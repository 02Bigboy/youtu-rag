"""
SMG Autonomous Module - 自主循环代码生成

核心改进：
1. 参考ADO提取的operator序列（作为指导）
2. 使用LLM自主循环，让LLM自己决定何时[THINK]/[CODE]/[Final Answer]
3. 最大10轮迭代，超时强制结束
4. 充分利用LLM的推理和规划能力
"""

import time
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.core.dtr_structures import (
    Operator, ExecutionPath, TableState, SMGNode, RewardVector
)


class SMGAutonomousModule:
    """
    自主循环SMG模块
    
    核心思想：
    - ADO提供operator序列作为参考（而非强制执行）
    - LLM自主决策每一步：思考、写代码、或输出答案
    - 每次可以输出[THINK]/[CODE]/[Final Answer]标识
    - 最多10轮迭代
    """
    
    def __init__(self, llm_client, event_callback, reward_evaluator):
        self.llm_client = llm_client
        self.reward_evaluator = reward_evaluator
        self.event_callback = event_callback  # 添加事件回调
        self.memory: List[SMGNode] = []
        self.persistent_memory: Dict[str, List[SMGNode]] = {}
    
    def _emit_event(self, name: str, event_data: Dict[str, Any]):
        """发送事件到回调函数"""
        if self.event_callback:
            try:
                self.event_callback(name, event_data)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")
    
    def execute_with_autonomous_loop(
        self,
        operator_sequence: List[str],  # ADO提取的operator序列(作为参考)
        operator_pool: List[Operator],  # 完整的operator池
        dataframe: pd.DataFrame,
        user_query: str,
        table_metadata: Dict[str, Any],
        schema_result=None,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        自主循环执行
        
        流程：
        1. 构建初始context（包含operator序列作为参考）
        2. 进入自主循环（最多max_iterations轮）
        3. 每轮LLM可以：
           - [THINK]: 分析当前状态，规划下一步
           - [CODE]: 生成并执行代码
           - [Final Answer]: 输出最终答案，结束
        4. 达到上限后强制结束
        
        Args:
            operator_sequence: ADO提取的operator名称列表(作为参考指导)
            operator_pool: 完整的operator定义池
            dataframe: 输入数据
            user_query: 用户问题
            table_metadata: 表格元数据
            schema_result: Schema信息（可选）
            max_iterations: 最大迭代次数
        
        Returns:
            Dict with:
                - final_df: 最终结果DataFrame
                - final_answer: 自然语言答案
                - execution_trace: 执行轨迹
                - memory_nodes: SMG节点
                - iterations_used: 实际使用的迭代次数
        """
        
        plan_str = "\n\n"
        plan_str += f"🔄 AUTONOMOUS LOOP EXECUTION (max {max_iterations} iterations)\n"
        plan_str += f"📋 Reference Operator Sequence: {' → '.join(operator_sequence)}\n"
        plan_str += f"   (LLM can follow or deviate based on its judgment)\n"

        self._emit_event(
            name="excel_agent.plan.delta",
            event_data={
                "content": plan_str
            }
        )

        self._emit_event(
            name="excel_agent.plan.done",
            event_data={
                "content": "<plan_done>"
            }
        )
        
        # 构建operator信息字典
        operator_map = {op.name: op for op in operator_pool}
        
        # 构建初始prompt
        initial_prompt = self._build_initial_prompt(
            user_query=user_query,
            dataframe=dataframe,
            operator_sequence=operator_sequence,
            operator_map=operator_map,
            table_metadata=table_metadata,
            schema_result=schema_result
        )
        
        # 对话历史
        conversation_history = [initial_prompt]
        
        # 执行状态
        current_df = dataframe.copy()
        execution_trace = []
        code_executions = []  # 记录所有代码执行
        
        # 自主循环
        for iteration in range(max_iterations):

            self._emit_event(
                name="excel_agent.task.start",
                event_data={
                    "type": "reasoning",
                    "operation": f"Iteration {iteration + 1}/{max_iterations}",
                    "content": "<reasoning_start>"
                }
            )
            
            # 调用LLM (提升max_tokens以允许详细分析)
            current_input = self._format_conversation(conversation_history)
            try:
                response = self.llm_client.call_api(current_input, max_tokens=3072)  # 提升到3072，平衡质量和速度
            except Exception as e:
                print(f"❌ LLM call failed: {e}")
                break
            
            # 记录response
            conversation_history.append(f"\n## Assistant Response (Round {iteration + 1})\n{response}")
            
            # 解析response，检测标识
            action, content = self._parse_response_action(response)
            
            print(f"🎯 Detected action: {action}")
            
            if action == "FINAL_ANSWER":
                # 找到最终答案，结束循环
                final_answer = self._extract_final_answer(response)
                print(f"✅ Final answer reached at iteration {iteration + 1}")
                print(f"   Answer: {final_answer[:100]}...")

                self._emit_event(
                    name="excel_agent.task.done",
                    event_data={
                        "type": "final answer",
                        "operation": f"[{action}]",
                        "content": "Finished"
                    }
                )
                
                return {
                    "final_df": current_df,
                    "final_answer": final_answer,
                    "execution_trace": execution_trace,
                    "memory_nodes": self.memory,
                    "iterations_used": iteration + 1,
                    "code_executions": code_executions,
                    "success": True
                }
            
            elif action == "CODE":
                # 提取并执行代码
                code = self._extract_code_block(content)
                
                if not code or code.strip() == "pass":
                    print(f"⚠️  No valid code extracted, prompting LLM...")
                    feedback = "No code was extracted. Please provide valid Python code in [CODE] block."
                    conversation_history.append(f"\n## System Feedback\n{feedback}")
                    continue
                
                print(f"🔧 Executing code...")
                print(f"   Code preview: {code[:100]}...")

                self._emit_event(
                    name="excel_agent.task.delta",
                    event_data={
                        "type": "code_generation",
                        "operation": f"[{action}]",
                        "content": f"{code}",
                        "mode": "code",
                        "clean": True
                    }
                )
                
                # 执行代码
                start_time = time.time()
                exec_result = self._execute_code_safe(code, current_df)
                execution_time = time.time() - start_time
                
                success = exec_result["success"]
                error_msg = exec_result.get("error", "")
                
                # 记录执行
                code_executions.append({
                    "iteration": iteration + 1,
                    "code": code,
                    "success": success,
                    "error": error_msg,
                    "execution_time": execution_time
                })
                
                if success:
                    # 更新current_df
                    current_df = exec_result["dataframe"]
                    print(f"   ✅ Execution succeeded")
                    print(f"   Result shape: {current_df.shape}")
                    
                    # 构建成功反馈
                    feedback = self._build_success_feedback(exec_result, current_df)
                    conversation_history.append(feedback)
                    
                    # 添加到execution trace
                    execution_trace.append({
                        "iteration": iteration + 1,
                        "action": "CODE_EXECUTION",
                        "code": code,
                        "success": True,
                        "result_shape": current_df.shape
                    })

                    self._emit_event(
                        name="excel_agent.task.done",
                        event_data={
                            "type": "code_execution",
                            "operation": f"[{action}] | ✅ Execution Success",
                            "content": f"✅ Execution Success: (Shape: {current_df.shape if isinstance(current_df, pd.DataFrame) else 'N/A'})"
                        }
                    )

                    task_type = "code_execution"
                    operation = f"[{action}] | ✅ Execution Success"
                    
                else:
                    # 执行失败
                    print(f"   ❌ Execution failed: {error_msg[:100]}")
                    
                    # 构建错误反馈
                    feedback = self._build_error_feedback(exec_result)
                    conversation_history.append(feedback)
                    
                    # 添加到execution trace
                    execution_trace.append({
                        "iteration": iteration + 1,
                        "action": "CODE_EXECUTION",
                        "code": code,
                        "success": False,
                        "error": error_msg
                    })

                    self._emit_event(
                        name="excel_agent.task.done",
                        event_data={
                            "type": "code_execution",
                            "operation": f"{action} | ❌ Execution Failed:「{error_msg[:50]}...」",
                            "content": f"❌ Execution Failed: {error_msg}"
                        }
                    )

                    task_type = "code_execution"
                    operation = f"[{action}] | ❌ Execution Failed"
            
            elif action == "THINK":
                # LLM在思考，记录并继续
                print(f"💭 LLM is thinking/reflecting...")
                thought = self._extract_think_content(content)
                print(f"   Thought: {thought[:150]}...")

                self._emit_event(
                    name="excel_agent.task.delta",
                    event_data={
                        "type": "reflection",
                        "operation": f"[{action}]",
                        "content": f"{thought}",
                        "clean": True
                    }
                )

                task_type = "reflection"
                operation = f"[{action}]"
                
                # 记录思考
                execution_trace.append({
                    "iteration": iteration + 1,
                    "action": "THINK",
                    "content": thought
                })
                
                # 提示继续
                continuation = """
Good thinking! Based on your analysis, what's your next step?

You can:
- Use **[CODE]** to write and execute code
- Use **[THINK]** to continue analyzing
- Use **[Final Answer]** if you have the complete answer

What would you like to do?
"""
                conversation_history.append(continuation)
            
            else:
                # 没有明确标识，提醒使用标识
                print(f"⚠️  No clear action tag detected, reminding LLM...")

                self._emit_event(
                    name="excel_agent.task.delta",
                    event_data={
                        "type": "reflection",
                        "operation": f"[{action}]",
                        "content": f"⚠️  No clear action tag detected, reminding LLM...",
                        "clean": True
                    }
                )

                task_type = "reflection"
                operation = f"[{action}]"
                
                reminder = """
Please use one of these tags to indicate your action:

- **[THINK]** - Analyze the current situation and plan next steps
- **[CODE]** - Write Python/Pandas code to process data
- **[Final Answer]** - Provide your final answer to the question

What would you like to do next?
"""
                conversation_history.append(reminder)
            
            if "code" in task_type:
                pass
            else:
                self._emit_event(
                    name="excel_agent.task.done",
                    event_data={
                        "type": task_type,
                        "operation": operation,
                        "content": "<task_done>"
                    }
                )
        
        # 达到最大迭代次数，强制结束
        print(f"\n{'='*60}")
        print(f"⚠️  Reached maximum iterations ({max_iterations})")
        print(f"{'='*60}")
        print(f"Forcing final answer extraction...")

        self._emit_event(
            name="excel_agent.task.start",
            event_data={
                "type": "reflection",
                "operation": f"[FORCE FINAL ANSWER]",
                "content": f"⚠️  Reached maximum iterations ({max_iterations})\nForcing final answer extraction...",
                "clean": True
            }
        )
        
        final_answer = self._force_extract_answer(
            conversation_history=conversation_history,
            current_df=current_df,
            user_query=user_query,
            table_metadata=table_metadata
        )

        self._emit_event(
            name="excel_agent.task.delta",
            event_data={
                "type": "reflection",
                "operation": f"[FORCE FINAL ANSWER]",
                "content": f"{final_answer}",
                "clean": True
            }
        )
        
        return {
            "final_df": current_df,
            "final_answer": final_answer,
            "execution_trace": execution_trace,
            "memory_nodes": self.memory,
            "iterations_used": max_iterations,
            "code_executions": code_executions,
            "success": False,  # 超时被迫结束
            "reason": "max_iterations_reached"
        }
    
    def _build_initial_prompt(
        self,
        user_query: str,
        dataframe: pd.DataFrame,
        operator_sequence: List[str],
        operator_map: Dict[str, Operator],
        table_metadata: Dict[str, Any],
        schema_result=None
    ) -> str:
        """构建初始prompt"""
        
        # DataFrame信息
        df_preview = dataframe.head(20).to_string()  # 减少到20行
        df_shape = dataframe.shape
        df_columns = list(dataframe.columns)
        
        # Operator参考信息
        operator_reference = self._build_operator_reference(operator_sequence, operator_map)
        
        # Schema信息
        schema_hint = ""
        if schema_result and schema_result.selected_col_headers:
            schema_hint = f"""
## 🎯 Schema Information (Relevant Headers)

**Relevant Columns** ({len(schema_result.selected_col_headers)}):
{', '.join(schema_result.selected_col_headers[:20])}

💡 These columns were identified as most relevant to the query.
"""
        
        # 表格元数据
        meta_hint = ""
        if table_metadata:
            meta_info = table_metadata.get("meta_info") or table_metadata
            if meta_info and not meta_info.get("error"):
                meta_lines = []
                meta_lines.append("\n## 📋 Table Structure Information:")
                if "header_rows_skipped" in meta_info:
                    meta_lines.append(f"- Header rows skipped: {meta_info.get('header_rows_skipped', 0)}")
                    if meta_info.get('has_merged_cells'):
                        meta_lines.append("- ⚠️  Merged cells preprocessed and expanded")
                meta_lines.append("- DataFrame `df` is clean and ready to use")
                meta_hint = "\n".join(meta_lines)
        
        prompt = f"""# Autonomous Code Generation Task

You are solving a tabular data question using an **autonomous iterative process**.

## 🎯 Your Goal

Answer this question: **{user_query}**

## 📊 Available Data

**DataFrame Shape**: {df_shape[0]} rows × {df_shape[1]} columns
**Columns**: {df_columns}

**Data Preview (first 20 rows)**:
```
{df_preview}
```
{schema_hint}
{meta_hint}

## 💡 Reference Operator Sequence (from ADO)

The following operator sequence was suggested by our analysis module.
You can **follow** these steps or **deviate** based on your judgment.

{operator_reference}

⚠️ **Important**: This sequence is a REFERENCE, not a strict requirement.
You can:
- Follow these steps if they make sense
- Skip steps if unnecessary
- Add additional steps if needed
- Reorganize the order if beneficial

## 🏷️ Action Tags You Can Use

At each iteration, indicate your action using one of these tags:

### 1. **[THINK]** or **[REFLECT]**
When you need to:
- Analyze the current situation and data structure
- Develop your analytical reasoning
- Plan your approach or reflect on results
- Draw insights from data patterns

**Quality over brevity**: Take 5-8 sentences to think thoroughly when needed.
Focus on deep analysis rather than just describing steps.

Example:
```
[THINK]
The question asks for equity analysis across categories. Looking at the data preview, 
I can see the distribution is highly skewed. The Gini coefficient measures inequality 
from 0 (perfect equality) to 1 (maximum inequality). Based on the visible values, 
top categories dominate the total. I should calculate group totals, compute the Gini, 
and identify concentration patterns. This will provide comprehensive inequality insights.
```

### 2. **[CODE]** (Optional - use only when truly needed)
Execute Python/Pandas code ONLY when:
- You need to process the full dataset (not just preview)
- Complex calculations are required
- Verification of computation is necessary

**Many questions can be answered through reasoning alone - code is not mandatory!**

Example:
```
[CODE]
```python
# Filter data
df = df[df['Year'] > 2020]
# Calculate sum
df = df.groupby('Category')['Value'].sum().reset_index()
```
```

**Critical code rules**:
- Use variable name `df` (already defined)
- Result MUST be assigned back to `df` as a DataFrame
- Use exact column names from the data
- Define all variables before use
- Use `round()` instead of format specifiers ({{:.2f}})

### 3. **[Final Answer]**
When you have the complete answer, provide a **detailed, well-structured response**.

**CRITICAL - Your final answer MUST follow the question's output format requirements**:
1. Use Markdown formatting (headers ##/###, lists, emphasis)
2. Present data in Markdown tables when appropriate
3. Include specific numerical results with proper context
4. Provide deep analysis and insights (not just numbers)
5. Give actionable, specific recommendations
6. For visualization questions: include complete Python code in ```python blocks

**Quality checklist for your final answer**:
- ✅ Comprehensive analysis (not superficial)
- ✅ Specific numerical evidence
- ✅ Interpretable insights and patterns
- ✅ Actionable recommendations (not vague suggestions)
- ✅ Professional formatting and structure

Example of HIGH-QUALITY final answer:
```
[Final Answer]

## Analysis Results

Based on comprehensive analysis of the dataset (N=500), here are the key findings:

### Summary Statistics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean | 45.2 | Above industry average |
| Std Dev | 12.3 | Moderate variability |

### Key Insights
1. **Trend Analysis**: The data shows a 23% increase over the period, indicating...
2. **Group Comparison**: Category A outperforms B by 2.5x (Cohen's d=0.82, large effect)

### Recommendations
1. **Prioritize Category A**: Given the strong performance and low variance, allocate 60% resources here
2. **Investigate Category C**: The declining trend (-15% YoY) requires immediate attention
3. **Optimize timing**: Peak occurs in Q2, suggesting seasonal strategy adjustment

[Include visualization code if requested]
```

## ⚠️ OUTPUT FORMAT CONSTRAINTS

**CRITICAL**: Each iteration, you MUST output EXACTLY ONE action tag and its content. 

**Rules**:
1. Start your response directly with one of: `[THINK]`, `[CODE]`, or `[Final Answer]`
2. Do NOT add any extra text or explanation before the action tag
3. Do NOT add any extra text or explanation after the action content
4. Output ONLY the selected action and nothing else

**Correct format**:
```
[THINK]
<your reasoning>
```

**Incorrect format** (DO NOT do this):
```
Let me analyze this first.
[THINK]
<your reasoning>
I will code next.
```

## 🚀 Start Your Analysis

You have up to 10 iterations. Think carefully and decide your approach.

**Available Actions**:
- **[THINK]**: Deep analytical reasoning (5-8 sentences for complex problems)
- **[CODE]**: Execute Python code (optional - only when computation is truly needed)
- **[Final Answer]**: Provide comprehensive, well-formatted final answer

**Guidelines**:
- **Prioritize quality over speed**: Thorough analysis beats quick responses
- **Code is optional**: Many questions can be answered through reasoning alone
- **For simple questions**: You can directly provide [Final Answer] if confident
- **For analytical questions**: Use [THINK] to develop deep insights before answering
- **For complex calculations**: Use [CODE] only when necessary to process full dataset
- **Final answers must be detailed**: Include statistics, insights, and specific recommendations

**Remember**: Your goal is to provide high-quality, comprehensive answers that demonstrate deep understanding.

Begin now:
"""
        
        return prompt
    
    def _build_operator_reference(
        self,
        operator_sequence: List[str],
        operator_map: Dict[str, Operator]
    ) -> str:
        """构建operator参考信息"""
        
        lines = []
        lines.append("**Suggested Steps**:")
        
        for idx, op_name in enumerate(operator_sequence, 1):
            operator = operator_map.get(op_name)
            if operator:
                lines.append(f"\n{idx}. **{operator.name}**")
                lines.append(f"   Description: {operator.description}")
                lines.append(f"   Category: {operator.category.value}")
            else:
                lines.append(f"\n{idx}. **{op_name}** (details not available)")
        
        return "\n".join(lines)
    
    def _format_conversation(self, history: List[str]) -> str:
        """
        格式化对话历史
        为了控制prompt长度和加快生成，只保留最近的关键轮次
        """
        if len(history) <= 6:
            # 少于6条消息，全部保留
            return "\n\n".join(history)
        else:
            # 保留初始prompt + 最近5轮对话
            initial_prompt = history[0]
            recent_history = history[-5:]
            return "\n\n".join([initial_prompt] + recent_history)
    
    def _parse_response_action(self, response: str) -> Tuple[str, str]:
        """
        解析response，识别action
        
        Returns:
            (action_type, content)
            action_type: "THINK", "CODE", "FINAL_ANSWER", "UNKNOWN"
        """
        
        response_lower = response.lower()
        
        # 检测[Final Answer]
        if "[final answer]" in response_lower:
            return ("FINAL_ANSWER", response)
        
        # 检测[CODE]
        if "[code]" in response_lower:
            return ("CODE", response)
        
        # 检测[THINK]或[REFLECT]
        if "[think]" in response_lower or "[reflect]" in response_lower:
            return ("THINK", response)
        
        return ("UNKNOWN", response)
    
    def _extract_code_block(self, response: str) -> str:
        """从response中提取代码块"""
        
        # 方法1: 查找[CODE]标签后的```python代码块
        pattern = r'\[CODE\]\s*```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # 方法2: 查找任意```python代码块
        pattern2 = r'```(?:python)?\s*(.*?)```'
        matches2 = re.findall(pattern2, response, re.DOTALL)
        
        if matches2:
            # 如果有多个代码块，合并它们
            return '\n\n'.join(m.strip() for m in matches2)
        
        # 方法3: 查找[CODE]后到下一个标签之间的内容
        pattern3 = r'\[CODE\](.*?)(?:\[THINK\]|\[REFLECT\]|\[Final Answer\]|$)'
        matches3 = re.findall(pattern3, response, re.DOTALL | re.IGNORECASE)
        
        if matches3:
            code = matches3[0].strip()
            # 移除可能的markdown标记
            code = re.sub(r'^```(?:python)?\s*', '', code)
            code = re.sub(r'```\s*$', '', code)
            return code.strip()
        
        return ""
    
    def _extract_think_content(self, response: str) -> str:
        """提取[THINK]标签的内容"""
        
        pattern = r'\[THINK\](.*?)(?:\[CODE\]|\[Final Answer\]|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        pattern2 = r'\[REFLECT\](.*?)(?:\[CODE\]|\[Final Answer\]|$)'
        matches2 = re.findall(pattern2, response, re.DOTALL | re.IGNORECASE)
        
        if matches2:
            return matches2[0].strip()
        
        return response[:500]  # 返回前500字符作为fallback
    
    def _extract_final_answer(self, response: str) -> str:
        """提取[Final Answer]内容"""
        
        pattern = r'\[Final Answer\]:?\s*(.*?)(?:\n\n\[|$)'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            answer = matches[0].strip()
            # 如果答案很短，可能提取不完整
            if len(answer) < 50:
                parts = response.split("[Final Answer]", 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    answer = re.sub(r'^:\s*', '', answer)
            
            return f"{answer}"
        
        # Fallback: 返回整个response
        return response
    
    def _execute_code_safe(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        """安全执行代码"""
        
        import numpy as np
        
        if not code:
            return {"success": False, "error": "Empty code"}
        
        # 安全检查
        forbidden = ["exit(", "quit(", "sys.exit", "os.system", "subprocess", 
                     "__import__", "eval(", "exec(", "open("]
        for kw in forbidden:
            if kw in code:
                return {"success": False, "error": f"Forbidden keyword: {kw}"}
        
        # 准备执行环境
        try:
            df_copy = df.copy()
        except:
            df_copy = df
        
        local_vars = {
            "df": df_copy,
            "pd": pd,
            "np": np
        }
        
        global_vars = {
            "pd": pd,
            "np": np,
            "__builtins__": __builtins__
        }
        
        # 执行
        try:
            exec(code, global_vars, local_vars)
            result_df = local_vars.get("df", df)
            
            # 自动转换dict为DataFrame
            if isinstance(result_df, dict):
                try:
                    result_df = pd.DataFrame(result_df)
                    print(f"    ℹ️  Auto-converted dict to DataFrame")
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Result is dict but cannot convert to DataFrame: {e}",
                        "dataframe": df
                    }
            
            # 确保是DataFrame
            if not isinstance(result_df, pd.DataFrame):
                return {
                    "success": False,
                    "error": f"Result must be DataFrame, got {type(result_df).__name__}",
                    "dataframe": df
                }
            
            return {
                "success": True,
                "dataframe": result_df,
                "error": None
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "dataframe": df
            }
    
    def _build_success_feedback(self, exec_result: Dict, result_df: pd.DataFrame) -> str:
        """构建成功执行的反馈"""
        
        preview = result_df.head(20).to_string()
        
        feedback = f"""
## ✅ Code Execution Successful!

**Result Summary**:
- Shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns
- Columns: {list(result_df.columns)}

**Result Preview (first 20 rows)**:
```
{preview}
```

---

**What's your next step?**
- Use **[THINK]** to deeply analyze these results and draw insights
- Use **[CODE]** if you need additional computation (optional)
- Use **[Final Answer]** if you can now provide a comprehensive, detailed answer
"""
        
        return feedback
    
    def _build_error_feedback(self, exec_result: Dict) -> str:
        """构建失败执行的反馈"""
        
        error_msg = exec_result.get("error", "Unknown error")
        
        feedback = f"""
## ❌ Code Execution Failed

**Error Message**:
```
{error_msg}
```

---

**Please use [THINK] to:**
1. Analyze what went wrong
2. Understand the root cause
3. Plan how to fix it

Then use **[CODE]** to try again with corrected code.
"""
        
        return feedback
    
    def _force_extract_answer(
        self,
        conversation_history: List[str],
        current_df: pd.DataFrame,
        user_query: str,
        table_metadata: Dict[str, Any] = None
    ) -> str:
        """达到最大迭代次数，强制提取答案"""
        
        # 尝试从最后几轮中提取有用信息
        recent_history = "\n\n".join(conversation_history[-5:])  # 增加到最后5轮
        
        # 构建表格metadata信息
        meta_info_str = ""
        if table_metadata:
            meta_info = table_metadata.get("meta_info") or table_metadata
            if meta_info and not meta_info.get("error"):
                meta_info_str = "\n## 📋 Table Metadata\n"
                if "header_rows_skipped" in meta_info:
                    meta_info_str += f"- Header rows skipped: {meta_info.get('header_rows_skipped', 0)}\n"
                if meta_info.get('has_merged_cells'):
                    meta_info_str += "- Merged cells were preprocessed\n"
                if "total_rows" in meta_info:
                    meta_info_str += f"- Original total rows: {meta_info.get('total_rows')}\n"
        
        # 构建强制提取prompt
        force_prompt = f"""
You've reached the iteration limit. Please provide your **COMPREHENSIVE final answer NOW** based on all the work done.

## Original Question
{user_query}

## Current DataFrame State
Shape: {current_df.shape}
Columns: {list(current_df.columns)}
{meta_info_str}

Preview (first 20 rows):
{current_df.head(20).to_string()}

## Recent History (last 5 interactions)
{recent_history[:3000]}

---

**CRITICAL: Provide a HIGH-QUALITY [Final Answer] that includes:**

1. **Specific numerical results**: Include all relevant statistics, calculations, and metrics
2. **Deep analysis**: Explain patterns, trends, and what the numbers mean
3. **Clear insights**: What are the key takeaways and implications?
4. **Actionable recommendations**: Specific, feasible suggestions (not vague advice)
5. **Professional formatting**: Use Markdown headers, tables, lists appropriately
6. **Visualization code**: If the question asks for charts, include complete Python code

**Your answer should demonstrate:**
- Thoroughness (comprehensive coverage of all aspects)
- Depth (insightful analysis, not just surface-level description)
- Clarity (well-organized, easy to understand)
- Utility (actionable and practical)

Use this format:
[Final Answer]
<your detailed, comprehensive answer here>
"""
        
        try:
            response = self.llm_client.call_api(force_prompt, max_tokens=4096)
            return self._extract_final_answer(response)
        except Exception as e:
            print(f"❌ Force extraction failed: {e}")
            # Fallback: 基于DataFrame生成简单答案
            if current_df.empty:
                return "[Final Answer]: No data available to answer the question."
            else:
                return f"[Final Answer]: Based on the processed data (shape: {current_df.shape}), here are the results:\n{current_df.head(10).to_string()}"
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        if not self.memory:
            return {
                "total_nodes": 0,
                "success_rate": 0.0,
                "avg_reward": 0.0
            }
        
        success_count = sum(1 for node in self.memory if node.success)
        
        return {
            "total_nodes": len(self.memory),
            "success_count": success_count,
            "failure_count": len(self.memory) - success_count,
            "success_rate": success_count / len(self.memory) if self.memory else 0.0
        }
    
    def clear_memory(self):
        """清空内存"""
        self.memory = []
