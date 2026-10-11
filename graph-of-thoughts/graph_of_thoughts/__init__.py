# 为了兼容旧版 run_tot_on_gqa.py 脚本而创建的别名和虚拟类

# 1. 基础库导入
from .operations.graph_of_operations import GraphOfOperations
from .controller.controller import Controller

# 2. 创建别名
GraphOfThoughts = GraphOfOperations

# 3. 定义虚拟的 Agent 类
class ProposeAgent:
    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

class ValueAgent:
    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

class ProposeAgentGot:
    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

class ValueAgentGot:
    def __init__(self, llm_caller):
        self.llm_caller = llm_caller

# 4. 定义虚拟的其他类
class Select:
    pass

class Generate:
    pass

class Evaluate:
    pass

class State:
    pass

class Action:
    pass

# 5. 定义 TreeOfThoughts 类
class TreeOfThoughts:
    def __init__(self, lm, propose_agent_class, value_agent_class):
        self.lm = lm
        self.propose_agent_class = propose_agent_class
        self.value_agent_class = value_agent_class
        self.graph = GraphOfOperations()
        self.controller = Controller(lm)

    def solve(self, *args, **kwargs):
        # 尝试调用 controller 的 start_task 方法
        # 注意：这里需要根据实际情况调整，因为新的API可能完全不同
        # 这里我们只做最基础的模拟，使其不报错
        print("Warning: TreeOfThoughts.solve is using a stub implementation.")
        return {"status": "stub_called", "args": args, "kwargs": kwargs}

# 注意：GraphOfOperations (即 GraphOfThoughts) 的构造函数签名与 TreeOfThoughts 不同，
# 它不需要 agent 类。因此，这里的 GraphOfThoughts 构造函数也需要适配。
# 让我们将 GraphOfThoughts 也重新定义为一个类似的类。
class GraphOfThoughts:
    def __init__(self, lm, propose_agent_class, value_agent_class):
        self.lm = lm
        self.propose_agent_class = propose_agent_class
        self.value_agent_class = value_agent_class
        self.graph = GraphOfOperations()
        self.controller = Controller(lm)

    def solve(self, *args, **kwargs):
        print("Warning: GraphOfThoughts.solve is using a stub implementation.")
        return {"status": "stub_called", "args": args, "kwargs": kwargs}
