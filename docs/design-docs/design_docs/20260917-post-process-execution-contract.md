# Search PostProcess：执行契约与 DataFrame 结果处理

- 创建：2026-09-17；更新：2026-10-09
- 状态：普通 Search 执行已接入；存在未解决的 review 问题，尚未完成发布验证
- 范围：显式 `FunctionChainStagePostProcess`，不重构 L0/L1 来源管理或 L2/Hybrid 结果恢复
- 相关设计：[Function Chain API](20260624-function-chain-api.md)、[Hybrid Function Chain](20260818-hybrid-search-function-chain.md)

## 1. 功能与边界

PostProcess 在 Proxy 已完成候选选择和字段物化之后，处理即将返回的 hits。每个请求最多包含一条非空链，只允许 `map`、`sort`、`limit`，按声明顺序执行；算子可以重复出现。

首版固定以下语义：

- 只转换、重排或截取已有候选，不参与召回、不扩大候选窗口、不补足被截掉的行。
- 每个 query 独立处理，一个 DataFrame chunk 对应一个 query。
- 计算输出只作为链内临时列，不新增公共返回字段、虚拟 FieldID 或 protobuf 协议。
- `$id`、`$score` 只读；不允许覆盖 schema 字段、写 JSON/dynamic 路径或输出 `$highlight`。
- 任一阶段失败即请求失败，不返回部分结果，也不静默回退为未执行 PostProcess 的成功响应。

L0/L1、L2/Hybrid、Merge 及原有 Order By/Highlighter 流程不在本次重构范围内。PostProcess 复用通用输入规划和算子，并通过专用结果转换接口接入。

## 2. 执行顺序与输入规划

```text
现有普通 Search 流水线
  reduce → 可选 rerank → 必要的 requery / 字段对齐
      ↓
PostProcess
  完整结果导入 DataFrame → 按序 map/sort/limit → 直接导出
      ↓
end
  最终字段投影、统计整理、既有 score rounding → 响应
```

`postProcessNode` 接收并输出现有逻辑 `result`，位于最终候选字段物化之后、`end` 之前。未配置 PostProcess 的请求继续走原有流水线。

`PostProcessPlan` 保存链及共用的 `DataFrameInputPlan`：

- 支持普通标量、TEXT，以及显式类型化的 JSON/dynamic 路径。
- 复用 `CompileDataFrameInputPlan`，不实现独立的路径解析或字段推断。
- JSON path 必须声明 Bool、Int64、Double 或 VarChar 类型提示；同一逻辑路径的类型提示不得冲突。
- dynamic 输入必须显式写 `$meta[...]`；未知 bare name 不自动回退为 dynamic key。
- 路径语法沿用共用解析器，包括其支持的嵌套键、数组索引和引号形式。完整 JSON root 及非 JSON 字段的 nested path 不作为计算输入。
- 多个路径共享物理 root 时，取字段依赖去重；前序 Map 产生的临时列不作为 collection 字段获取。
- 缺失、null、不兼容值及数值转换失败投影为 typed NULL；损坏的 JSON 返回数据完整性错误。

隐藏输入加入搜索取字段或 requery 依赖，但不能因此自动进入客户端输出。TEXT 输入需要 requery；无法完成该物化的模式必须显式拒绝。

PostProcess 读取的 `$score` 是上游实际交付的分数，可能经过 rerank 或 rounding。它不恢复 ANN 原始分数，也不自行转换 metric；既有 `end` 的响应 rounding 仍在其后执行。

## 3. 排序、截断与分页

令 `Cq` 为现有流水线交付给第 q 个 query 的有序候选，结果为：

```text
Rq = opN(...op2(op1(Cq)))
```

| 算子 | 行数与顺序 | 规则 |
|---|---|---|
| Map | 保持行数、行序和 chunk 边界 | 校验函数输出数量、类型及 shape；普通临时列可被后续算子读取或覆盖 |
| Sort | 保持行数，改变 chunk 内顺序 | 根据排序键生成临时排列，再重排 DataFrame 全部列 |
| Limit | 减少或保持行数，保留相对顺序 | 仅截取当前位置的候选，多个 Limit 依次生效 |

Sort 支持当前可比较的 Bool、数值、String 类型。嵌套数据可以随行移动，但不因此成为合法排序键。

- 新 `orders` 格式支持多个键、独立 ASC/DESC 和 NULL 顺序；默认 ASC NULLS LAST、DESC NULLS FIRST、稳定排序，不隐式追加 `$id`。
- 旧 `desc` 格式保持 NULLS LAST 和默认 `$id` 升序 tie-break。
- 相等键的稳定排序只保留进入本次 Sort 的顺序，不保证跨请求的候选初始顺序相同。

请求的 topK/offset 仍由上游原有逻辑处理，PostProcess 不重新应用，也不自动添加 Sort/Limit。链内 `limit(L, O)` 要求 `L > 0`、`O >= 0`，数学语义为取当前序列的 `[min(O,n), min(O+L,n))`；极端参数和整数溢出属于验收边界。

| 示例 | 结果 |
|---|---|
| 上游已分页得到 `[C,D,E]`，排序得到 `[E,C,D]` | 返回该排序，不再次应用请求 offset |
| 随后执行 `limit(2,1)` | `[C,D]`，这是链内显式截断 |
| `[A,B,C,D]` 先 `limit(2)` 再排序 | 只能重排 A/B，不能找回 C/D |
| `[A,B,C]` 执行 `limit(2,10)` | 空结果，仍保留该 query 的位置 |

## 4. 完整 DataFrame 与直接导出

PostProcess 不创建 `$row_id`，不在执行后按主键或额外行号回查原始响应。返回字段的实际值进入 DataFrame，与计算列一起经过 Sort/Limit。

DataFrame 承载三类内容：

1. **计算列**：`$id`、`$score`、输入规划得到的逻辑列及临时 Map 输出。
2. **响应列**：使用私有 `$result_field:<fieldID>:<name>` 名称携带原始返回值，避免临时变量覆盖实体数据。
3. **附带信息**：Distances 是逐行列；Recalls 按其构造端语义为每 query 一个值，保存在元数据中，不随行重排。

`DataFrame.resultFields` 按需保存空的 `FieldData` 描述，包括原始名称、类型、FieldID、维度、数组元素类型和 dynamic 标记。它只保存 schema 信息，不保存行数据或映射表；`CopyFieldMetadata` 在新建 DataFrame 时传递这些描述。

| 原始字段 | Arrow 表示 |
|---|---|
| 普通标量、TEXT、Timestamptz | 现有标量 Arrow 类型 |
| JSON、Geometry WKB、SparseFloatVector | Binary，保存实际值 |
| Geometry WKT | String |
| FloatVector | FixedSizeList<Float32> |
| Binary/Float16/BFloat16/Int8 vector | FixedSizeBinary |
| Array、ArrayOfVector | Binary，保存每行嵌套 ScalarField/VectorField 的 protobuf 值及其元素 NULL 信息，不保存行引用 |

nullable vector 的紧凑 protobuf 输入展开为带 NULL 的 Arrow 行，导出时重新生成紧凑 payload。响应列描述随列传递，避免丢失类型、维度或 NULL 语义。

`FromSearchResultDataWithPayload` / `ToSearchResultDataWithPayload` 由 PostProcess 显式调用。已有转换接口和其他 chain 的数据承载方式不变；本功能不需要给 Merge 增加 payload 支持。

## 5. 输出协议

首版沿用现有 Search protobuf 和 SDK 返回结构：

| 内容 | 契约 |
|---|---|
| NumQueries、Topks | 保留全部 query，包括空 query；Topks 为各 query 最终实际行数 |
| TopK | `max(Topks)`，全部为空时为 0 |
| IDs、scores、Distances | query-major 展平，所有逐行数据使用相同顺序 |
| FieldsData | 只返回原有输出投影允许的字段，保留物理字段身份、类型和 NULL 语义 |
| Recalls、统计与成本 | 按 query 或既有非逐行语义保留，不替换为剩余 hit 数 |
| 临时列、逻辑路径列、私有列名 | 不返回，不分配虚拟 FieldID，不写入 `$meta` |
| HighlightResults | 显式 PostProcess 首版不生成；旧 highlighter 保持原路径 |

`output_fields` 或 `*` 不会把普通计算列变为返回字段。临时列与真实 dynamic key 同名时，客户端仍应获得 collection 中的原始 dynamic 值。

为计算获取完整 `$meta` 时，返回前必须裁剪为用户请求的 dynamic keys。该要求尚存在一个实现缺口，见第 7 节。

零命中也必须校验链和参数。所有行被 Limit 截掉后，保留长度为 NumQueries 的零 Topks 数组；隐藏依赖不能泄漏到空结果的字段描述中。

## 6. 首版组合限制与错误处理

| 组合 | 当前服务端范围 |
|---|---|
| 普通 Search，有/无 requery | 支持 |
| 普通 Search + 既有合法 L0/L1/L2 或 function_score + PostProcess | 支持；不放开其他 rerank 来源之间原有的互斥规则 |
| 显式 PostProcess + order_by_fields / 旧 highlighter | 拒绝 |
| Hybrid Search，包括 sub-search PostProcess | 拒绝 |
| search_aggregation、Iterator V1/V2、Search group-by | 拒绝 |
| 以 ArrayOfVector 为 ANN 目标的搜索 | 拒绝；不等于禁止携带 ArrayOfVector 返回字段 |
| namespace partition 模式 + TEXT 输入 | 拒绝，该模式跳过所需 requery |
| dynamic 写回、覆盖 schema 字段、`$highlight` 输出 | 拒绝 |

用户配置的路径、类型提示、参数或组合错误属于 InputError。内部结果缺列、类型或 shape 违约属于系统错误；函数输出违约沿用 FunctionFailed，损坏 JSON 沿用 DataIntegrity。添加错误上下文需保留原错误码，不能把取消、超时或资源失败改为数据损坏。

请求 context 贯穿执行；成功、失败和取消路径均需释放 Arrow 中间结果。

## 7. 实现状态、已知问题与验收

代码已迁入 Master 的 `internal/proxy/dql` 布局，普通 Search 的 PostProcess 执行节点已接入。当前状态不等于发布验证完成。

**已知问题：动态字段投影可能跳过。** `projectPostProcessDynamicFields` 当前同时检查 `$meta` 名称与 `IsDynamic`。非 requery 路径中，`fillFieldNames` 只补名称，`endOperator` 在 PostProcess 之后才设置该标记，因此隐藏计算 key 可能留在原始响应 payload 中。需要按 schema 识别动态根或提前补齐标记，并使用 `IsDynamic=false` 的真实 QueryNode 形态补回归；已有预置标记的测试不能证明此路径正确。

**优化模式待核对。** 当前 PostProcess 强制 `SearchType_DEFAULT`，属于保守限制，尚未逐项证明其他优化模式都不兼容。

验证记录按版本区分：

- 迁移前的范围收缩版本：PostProcess/chain 定向 Go 回归通过；更早版本运行过真实 SDK smoke，不能直接视为当前分支已通过。
- 当前 Master 迁移后：格式、冲突和 `chain/types` 检查通过。PostProcess/chain 测试被过旧的 native 头文件阻塞：缺少 JSON stats executor 接口，packed reader 声明与源码签名不一致，需要重建依赖后重跑。
- 历史全量 `make test-go` 曾被未修改的 `storage/TypesTest.cpp` 的 constexpr/consteval 编译错误阻塞；不能据此声称全量 Go 测试通过。
- StorageV3 sealed、性能、混合版本和当前分支端到端验证未完成。历史本地存储还出现过 bloom-filter/JSON stats 路径问题，应与 PostProcess 功能失败分开记录。

发布前至少覆盖：

1. 有/无 rerank × 有/无 requery 四条普通 Search 路径，以及 PostProcess 同既有阶段组合。
2. 请求分页与多个链内 Limit、空结果、多 NQ、NULL、多键方向和稳定排序。
3. 普通字段、TEXT、typed JSON/dynamic 输入、临时依赖及共享 root；验证隐藏 key 不泄漏，尤其是未设置 IsDynamic 的 QueryNode 输入。
4. IDs、scores、向量、数组、JSON、nullable 数据逐行对齐；导入后不依赖原始响应仍可排序和导出。
5. 参数错误、JSON 损坏、缺列/错 shape、函数失败、取消/超时的来源到响应路径，以及资源释放和无部分成功。
6. 禁止组合明确失败；未配置 PostProcess 的请求行为不变。

SDK smoke 使用 [post_process_smoke.py](../../../tests/scripts/post_process_smoke.py)，只创建并删除自己的随机命名 collection：

```bash
POST_PROCESS_URI=http://127.0.0.1:39530 python tests/scripts/post_process_smoke.py

# 要求服务端启用 common.storage.useLoonFFI；仅验证 growing TEXT。
POST_PROCESS_URI=http://127.0.0.1:39530 POST_PROCESS_TEXT=1 POST_PROCESS_SEALED=0 \
  python tests/scripts/post_process_smoke.py
```

类型化路径用例通过测试用 EncodedChain 编码现有 `$input_data_types` 参数，不代表已新增公共 SDK API。SDK 的类型化列便捷接口及 ranker/function_chains 组合校验需单独确认，不能由服务端能力推导。

## 8. 代码入口与后续范围

| 职责 | 入口 |
|---|---|
| 链校验、首版限制 | [function_chain_validator.go](../../../internal/proxy/dql/function_chain_validator.go) |
| 输入规划 | [post_process_plan.go](../../../internal/proxy/dql/post_process_plan.go)、[input_plan.go](../../../internal/util/function/chain/input_plan.go) |
| 搜索依赖、节点接入 | [task_search.go](../../../internal/proxy/dql/task_search.go)、[search_pipeline.go](../../../internal/proxy/dql/search_pipeline.go) |
| PostProcess 执行及动态投影 | [post_process_operator.go](../../../internal/proxy/dql/post_process_operator.go) |
| 完整结果转换、字段校验 | [result_payload.go](../../../internal/util/function/chain/result_payload.go)、[result_field_validation.go](../../../internal/util/function/chain/result_field_validation.go) |
| DataFrame、算子执行 | [dataframe.go](../../../internal/util/function/chain/dataframe.go)、[chain.go](../../../internal/util/function/chain/chain.go) |
| JSON 路径物化 | [json_projector.go](../../../internal/util/function/chain/json_projector.go) |

后续独立设计包括计算字段显式返回、dynamic 写回、HighlightExpr、Iterator continuation、Hybrid、group-by/element-level，以及旧 Order By/Highlighter 迁移。这些不属于当前首版承诺。
