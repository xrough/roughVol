# 模拟投行级期权/期货系统框架设计

## 1. 目标

构建一个模块化、可扩展、可回测、可模拟真实投行流程的衍生品交易系统，支持：

* 多资产（期权、期货等）
* 买方与卖方两类业务逻辑
* 策略插件化开发
* 回测 / 模拟 / 实盘统一接口
* 投行级风险与流程控制

---

## 2. 总体架构

### 分层结构

```
Application Layer
    ↓
Coordinator / Workflow Layer
    ↓
Adapters Layer
    ↓
Core Domain Layer
    ↓
Infrastructure Layer
```

### 各层职责

#### 2.1 Core Domain Layer（核心领域层）

定义稳定、抽象、与业务逻辑强绑定的金融概念：

* Instrument（Option / Future / Underlying）
* Price / OrderBook / Trade
* Position
* Risk Metrics（Greeks、PnL、VaR）
* MarketState
* Event 基础结构

要求：

* 不依赖外部系统
* 严格类型与接口定义
* 支持高频与低频数据统一表示

示例：

```
PriceProvider (abstract)
    ├── HFPriceProvider
    ├── LFPriceProvider
```

---

#### 2.2 Adapters Layer（适配层）

隔离核心与外部系统。

类型：

* Price Adapter（数据库 / 行情接口）
* Execution Adapter（交易所 / 模拟撮合）
* Strategy Adapter（交易员插件）
* Risk Adapter
* Logging Adapter
* Market Simulator Adapter

特点：

* 统一接口
* 支持实时 / 回测模式切换
* 支持日志与校验

---

#### 2.3 Coordinator / Workflow Layer（协调层）

负责流程编排，不实现业务逻辑。

示例流程：

* 回测流程
* 实盘交易流程
* 对冲流程
* 风险检查流程

职责：

* 调用各 Adapter
* 管理状态
* 处理异常

---

#### 2.4 Application Layer（应用层）

面向用户：

* 交易员界面
* 策略配置
* 实验管理
* 权限控制

应用层只调用 Workflow，不直接操作 Core。

---

## 3. Event-Driven 架构

### 3.1 Event 定义

事件 = 发生的事实，而非业务逻辑。

结构示例：

```
Event {
    id
    type
    timestamp
    payload
}
```

示例事件：

* PriceUpdated
* OrderSubmitted
* TradeExecuted
* RiskLimitBreached
* StrategySignalGenerated

### 3.2 Event 机制

* 发布 / 订阅（Pub/Sub）
* 异步处理
* 唯一 ID 追踪结果
* 事件版本化（禁止破坏旧事件）

作用：

* 模块解耦
* 支持回放
* 支持审计

---

## 4. 插件化 Strategy 设计

### 4.1 Strategy Interface

```
StrategyPlugin {
    generate_signal()
    risk_constraints()
    on_event()
}
```

插件特点：

* 独立开发
* 动态加载
* 支持回测 / 模拟 / 实盘

交易员只需实现接口。

---

## 5. 买方 vs 卖方的划分

### 5.1 不在最底层区分

Core Domain Layer 应保持中立。
Instrument、Price、Risk 等概念对双方通用。

### 5.2 在 Strategy / Workflow 层区分

#### 买方目标

* 接受报价
* 优化执行
* 管理组合风险
* 追求收益最大化

典型插件：

* Alpha Strategy
* Portfolio Optimization
* Execution Algo

终点：
**生成投资收益并控制风险**

---

#### 卖方目标

* 生成报价
* 管理库存风险
* 对冲风险
* 提供流动性

典型插件：

* Quote Engine
* Inventory Management
* Hedging Engine

终点：
**赚取 Bid-Ask Spread 并保持风险中性**

---

### 5.3 为什么不在 Core 区分

因为：

* Position、Price、Risk 对双方相同
* 区分应在行为层（Strategy）而非数据层

---

## 6. 市场模拟（Market Simulation）

### 6.1 层级

1. 简单模型

   * 成交概率模型
   * 基于历史成交量

2. Order Book 模拟

   * 排队机制
   * 撮合规则
   * 延迟模拟

3. Agent-Based Market

   * 多策略参与者
   * 动态流动性

### 6.2 可借鉴开源项目

* QuantLib（定价）
* Backtrader / Zipline（回测）
* OrderBook 模拟引擎

---

## 7. 日志与审计

日志必须记录：

* 请求参数
* 数据源
* 结果
* 延迟
* 错误

支持：

* Debug / Info / Warning / Error
* 可追踪 Event ID
* 可审计交易流程

---

## 8. 投行级扩展需求

* 并发处理
* 权限系统
* 风控 Gate
* 历史回放
* 多用户策略实验
* 可解释 AI
* 云原生部署

---

## 9. 最终流程示例

### 买方

```
Strategy → Signal Event → Execution Adapter → Trade Event → Risk Check → Position Update
```

### 卖方

```
Market Data → Quote Engine → Quote Event → Client Trade → Inventory Update → Hedging Strategy
```

---

## 10. 设计原则总结

* Core 稳定、Adapter 灵活
* Workflow 编排流程
* Event 解耦模块
* Strategy 插件化
* 买卖方在行为层区分
* 回测 / 实盘统一接口
* 日志与审计优先

---

如果需要，可以继续细化：

* Core Price 数据结构
* Event Schema 设计
* Strategy Plugin 示例代码
* OrderBook 模拟算法
