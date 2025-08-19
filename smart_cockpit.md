# Potential Topics
1. Multimodal human-machine interaction (Speech + visual)
    - NLU + speech control
    - 

2. Virtual assistant
    - On vehicle virtual assistant/copilot
    - Multi-turn dialogue, TOD planning

3. LLM on-device deploy
    - LLM distillation, quantization 
    - Multilingual deploy

4. Personalization recommandation
    - Based on behaviour history, user command to infer user's intention


# Potential Questions

1. LLM in vehicle
    - How does LLM used on vehicle? 
    - How to do on-vehicle inference optimization?
    - How to deal with latency, safty and compuational cost?

Answer: 目前车端语音助手仍然以命令式为主，缺乏理解和上下文记忆，LLM可以提升自然交互体验。但车端对时延、算力、功耗要求苛刻，因此需要做压缩和裁剪，比如使用int8量化后的Qwen-7B或者采用MiniCPM、Gemma等模型，甚至自己训练LoRA Adapter实现定制对话。


2. Are you familiar with LLM?
   - Did you do SFT? What did you use（LoRA / PEFT / P-Tuning?
   - Have you used RAG?
   - Do you have experience with multiturn dialogue, memory, intention recognition?

3. Edge deploy?
   - Could side or edge deploy? 
   - Traditional pipeline or End-to-end?

Answer: 在边缘设备上，内嵌Prompt + LoRA微调是更高效的组合。理想方案是结合两者：静态任务靠Prompt，动态知识靠RAG，甚至未来做一个任务分流机制。