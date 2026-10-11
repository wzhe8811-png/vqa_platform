import json
import os
from PIL import Image
import time
import argparse
from typing import List

# --- 修改导入 ---
from tot_got_integration import VisionToTextReasoner

def _create_and_run_got_solver():
    """
    这个函数封装了所有与 graph-of-thoughts 相关的导入和类定义，
    以避免在脚本顶层导入失败的问题。
    """
    # 尝试从 graph_of_thoughts 的真实子模块导入
    try:
        from graph_of_thoughts.operations.graph_of_operations import GraphOfOperations
        from graph_of_thoughts.controller.controller import Controller
        from graph_of_thoughts.operations.operations import Generate, Selector, Score as Evaluate
        from graph_of_thoughts.prompter import Prompter
        from graph_of_thoughts.parser import Parser
        from graph_of_thoughts.operations.thought import Thought
        from graph_of_thoughts.language_models.abstract_language_model import AbstractLanguageModel
        print("Successfully imported from graph_of_thoughts package.")
    except ImportError as e:
        print(f"Failed to import from graph_of_thoughts package: {e}")
        raise e # 如果连真正的模块都找不到，那就直接报错，不要尝试其他路径了

    # 定义一个简单的 Prompter 实现，必须实现所有抽象方法
    class SimplePrompter(Prompter):
        def generate_prompt(self, state, **kwargs):
            return f"Given the current state: '{state}'. Please generate a concise next thought or step towards solving the problem."

        def score_prompt(self, state, **kwargs):
            return f"On a scale of 1-10, how likely is this thought/state to lead to a correct solution? Rate it and briefly justify: '{state}'."

        def validation_prompt(self, state, **kwargs):
            return f"Is this thought/state valid and logically consistent? Answer yes or no, and explain why: '{state}'."

        def improve_prompt(self, state, **kwargs):
            return f"The previous thought was: '{state}'. Can you refine or improve it?"

        def aggregation_prompt(self, states, **kwargs):
            return f"You have several thoughts/states: {states}. Please combine them into a single, coherent next step or conclusion."

    # 定义一个简单的 Parser 实现，必须实现所有抽象方法
    class SimpleParser(Parser):
        def parse_generate_answer(self, answer, **kwargs):
            # 解析 "generate" 操作的回复，返回一个状态列表
            return [answer.strip()]

        def parse_score_answer(self, answer, **kwargs):
            # 解析 "score" 操作的回复，通常返回一个数字分数
            # 这里简化处理，提取第一个数字
            import re
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return float(numbers[0])
            return 0.0

        def parse_validation_answer(self, answer, **kwargs):
            # 解析 "validation" 操作的回复，返回布尔值
            return "yes" in answer.lower() or "valid" in answer.lower()

        def parse_improve_answer(self, answer, **kwargs):
            # 解析 "improve" 操作的回复，返回一个改进后的状态
            return [answer.strip()]

        def parse_aggregation_answer(self, answer, **kwargs):
            # 解析 "aggregation" 操作的回复，返回一个聚合后的状态
            return [answer.strip()]


    def select_top_n(n: int):
        """
        返回一个选择器函数，该函数从输入的Thought列表中选择前n个。
        这是为了适配 graph-of-thoughts 库中 Selector 类的接口。
        """
        def selector_func(thoughts: List[Thought]) -> List[Thought]:
            # thoughts 是一个 Thought 对象的列表
            # 按照它们的分数进行排序（如果有的话），然后返回前 n 个
            # 这里我们做一个简化的选择，直接返回前 n 个
            return thoughts[:n]
        return selector_func

    # 定义一个适配器，将我们的LLM调用函数包装成 AbstractLanguageModel 的子类
    class LLMAdapter(AbstractLanguageModel):
        def __init__(self, llm_caller_func, model_name="CustomLLM", config_path=""):
            # 调用父类构造函数，传入模型名称和一个空的配置路径
            super().__init__(config_path=config_path, model_name=model_name)
            self.llm_caller_func = llm_caller_func

        # 重写 load_config 方法，因为默认行为会寻找一个不存在的 config.json 文件
        def load_config(self, path: str) -> None:
            """
            重写此方法以避免因缺少 config.json 文件而崩溃。
            """
            # 不执行任何操作，或者可以在这里设置一些默认配置
            self.config = {} # 设置一个空的配置字典
            # self.logger.debug("Config not loaded, using empty default.")

        def query(self, query: str, num_responses: int = 1) -> List[str]:
            """
            实现 query 方法。
            对于我们的简单调用器，一次只返回一个响应。
            """
            # 根据 num_responses 的数量，多次调用
            responses = []
            for _ in range(num_responses):
                response = self.llm_caller_func(query)
                responses.append(response)
            return responses

        def get_response_texts(self, query_responses: List[str]) -> List[str]:
            """
            实现 get_response_texts 方法。
            因为我们 query 返回的就是字符串列表，所以这里直接返回。
            """
            # query_responses 已经是 List[str] 了
            return query_responses


    def solve_tot_for_gqa(llm_caller, visual_description, question_text, task="gqa"):
        """
        使用Graph of Operations (来自 graph-of-thoughts 库) 解决GQA问题
        """
        # 1. 创建操作图
        gop = GraphOfOperations()

        # 2. 定义 Prompter 和 Parser
        simple_prompter = SimplePrompter()
        simple_parser = SimpleParser()

        # 3. 创建 LLM 适配器
        adapted_llm = LLMAdapter(llm_caller)

        # 4. 定义问题参数
        initial_state = f"Visual Description: {visual_description}\nQuestion: {question_text}\nInitial Thought:"
        problem_params = {"initial_state": initial_state}

        # 5. 创建控制器
        # Controller 需要: lm (现在是 adapted_llm), graph, prompter, parser, problem_parameters
        ctrl = Controller(adapted_llm, gop, simple_prompter, simple_parser, problem_params)

        # 6. 构建操作图
        gop.append_operation(Generate()) # 添加一个生成操作
        gop.append_operation(Selector(select_top_n(1))) # 添加一个选择器操作，选择1个
        gop.append_operation(Evaluate()) # 添加一个评估操作

        # 7. 运行控制器
        try:
            ctrl.run() # 执行操作图
            final_thoughts = ctrl.get_final_thoughts() # 获取最终的思考结果
            # final_thoughts 是一个嵌套列表 [[Thought, ...], ...]
            # 我们取最后一个操作的最后一个想法作为中间结果
            if final_thoughts and final_thoughts[-1]:
                 # 取最后一个叶子节点的第一个（也是唯一一个）thought
                final_state = final_thoughts[-1][0].state
            else:
                final_state = "No solution found by GoT."
        except Exception as e:
            print(f"Error during GoT execution: {e}")
            final_state = "Execution failed."

        # 8. 提取最终答案
        extraction_prompt = f"Based on the following reasoning: '{final_state}', provide a concise final answer to the original question."
        final_answer = llm_caller(extraction_prompt)
        return final_answer
    
    # 将内部定义的函数返回出去，供外部调用
    return solve_tot_for_gqa

# --- 主程序逻辑 ---

def load_gqa_questions(json_file_path):
    """从JSON文件加载GQA问题"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查数据类型并相应处理
    if isinstance(data, dict):
        # GQA 数据集的格式通常是 { "question_id": { ... }, ... }
        # 将其转换为 [ { ... }, ... ] 的列表格式
        questions_list = list(data.values())
        print(f"Detected dictionary format with {len(questions_list)} entries.")
        return questions_list, True # Return flag indicating dict format
    elif isinstance(data, list):
        print(f"Detected list format with {len(data)} entries.")
        return data, False # Return flag indicating list format
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

def find_image_path(image_filename, input_dir):
    """
    在多个可能的位置查找图片文件。
    """
    # 定义可能的图片路径列表
    possible_paths = [
        # 1. 在 input_dir 下的 images 子目录 (旧版逻辑)
        os.path.join(input_dir, 'images', image_filename),
        # 2. 在 input_dir 下直接 (某些数据集格式)
        os.path.join(input_dir, image_filename),
        # 3. 在全局的 gqa_standard 图片库中
        os.path.join('./data/gqa_standard/images', image_filename),
    ]
    
    # 尝试在每个可能的路径下查找带扩展名的文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    for base_path in possible_paths:
        # 首先尝试不带扩展名的路径（以防文件名本身就包含扩展名）
        if os.path.exists(base_path):
            return base_path
            
        # 然后尝试加上各种扩展名
        name, ext = os.path.splitext(base_path)
        if not ext: # 如果原始路径没有扩展名
            for extension in image_extensions:
                full_path = name + extension
                if os.path.exists(full_path):
                    return full_path
        # 如果原始路径有扩展名，也直接尝试一次
        elif os.path.exists(base_path):
             return base_path
            
    return None # 如果都找不到，返回 None


def main():
    parser = argparse.ArgumentParser(description='Run ToT on GQA dataset.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing questions and images.')
    parser.add_argument('--output_file', type=str, required=True, help='File to save the results.')
    args = parser.parse_args()

    # 加载GQA模型和LLM
    print("正在加载本地LLaVA模型...")
    reasoner = VisionToTextReasoner(vision_model_path="/root/autodl-tmp/vision_qa_platform/llava-1.5-7b-hf",
                                 llm_model_path="/root/autodl-tmp/vision_qa_platform/text_model/qwen2_1_5b_instruct")
    print("LLaVA模型加载完成。")
    print("正在加载本地Qwen LLM...")
    print("Qwen LLM加载完成。")

    # 构建问题文件路径
    questions_file_path = os.path.join(args.input_dir, 'questions', 'testdev_balanced_questions.json')

    # 加载GQA问题
    print(f"正在从 {questions_file_path} 加载问题...")
    questions_data, is_dict_format = load_gqa_questions(questions_file_path)
    print(f"共加载了 {len(questions_data)} 个问题。")

    # --- 在这里调用内部函数，创建 solver ---
    print("正在创建 graph-of-thoughts 推理器...")
    solve_tot_for_gqa = _create_and_run_got_solver()
    print("graph-of-thoughts 推理器创建完成。")

    results = []

    # Determine data structure and iterate accordingly
    items_to_iterate = questions_data[:5] # Process first 5 for testing

    for i, item in enumerate(items_to_iterate):
        # Handle both dict and list formats by getting keys safely
        if is_dict_format:
            # If it was originally a dict, 'item' is now the value.
            # The original key (question_id) is lost in the conversion to list.
            # We'll use the 'imageId' or a generated key as the id.
            question_id = item.get('question_id', item.get('imageId', f'dict_item_{i}'))
        else:
            # If it was a list, 'item' is the object itself.
            question_id = item.get('question_id', item.get('id', str(i)))

        question_text = item['question']
        # The image key might be 'imageId' or 'image'
        image_filename = item.get('imageId', item.get('image'))

        # Use the new function to find the image
        image_path = find_image_path(image_filename, args.input_dir)

        if image_path is None:
            print(f"Warning: Image file not found for ID/name: {image_filename}. Skipping this question.")
            continue # Skip this question if no image is found

        print(f"\n--- 处理问题 {question_id} ---")
        print(f"问题: {question_text}")

        # 1. 获取视觉描述
        try:
            visual_desc = reasoner.get_visual_description(image_path, question_text)
            print(f"视觉描述: {visual_desc}")
        except Exception as e:
            print(f"获取视觉描述时出错: {e}")
            continue

        # 2. 使用GoT解决GQA问题
        try:
            # Pass the LLM caller from our reasoner to the GoT solver
            answer_pred = solve_tot_for_gqa(reasoner.get_llm_caller(), visual_desc, question_text)
            print(f"GoT预测答案: {answer_pred}")
        except Exception as e:
            print(f"解决GoT问题时出错: {e}")
            answer_pred = "error"

        # 记录结果
        result_entry = {
            "question_id": question_id,
            "question": question_text,
            "predicted_answer": answer_pred,
            "ground_truth": item.get('answer', 'unknown'), # Assuming 'answer' is the GT key
            "image_path": image_path
        }
        results.append(result_entry)

    # 保存结果
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果已保存到 {args.output_file}")


if __name__ == "__main__":
    main()
