# Full-Loop
# Paper coding: <Full-Loop by Fusion Knowledge, Fusion Scorers, and Entailment to Mitigate Hallucination in Large Language Models>

***I run the experiments with 3 models (Baselines and MyLoops) and 2 datasets. (select 50 questions) 
**Forgive me:
                      (time limited---because of the 'entailment' of human, it's a quite quite time-consuming task, and no better GPU at home)
*********************************************************************************************************
***************************************contributions****************************************************
1. Add 'Factuality evaluation' on these 3 models to mitigate hallucinations.
2. Do my improvements:
   
	Methods:

		(1) fusion knowledge with additional retrieved message
		(2) design a factuality scorer with the help of the GPT-3.5-turbo
	 	(3) fusion-scorer loop
		(4) entailment judge loop
	
	Experiments:

		(1) Hallucination analysis
		(2) 50 samples to evaluate the proposed methods
*********************************************************************************************************
*********************************************************************************************************
=======================================================================================
* ## Experiments Record:

1. # FastChat：vicuna-7b-v1.5
	## ------------dataset: [MASHQA]
		### [1]. MyLoop
		run the loop_new successfully. √
		[
		--eval_num=50
		--MAX_LOOP_1=3
		--MAX_LOOP_2=2
		--...more
		]
		the results are load in: 
		/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/MASHQA/=========
		--- history_Loop1.txt
		--- history_Loop2.txt
		--- judge_messages_in_Loop1_vicuna-7b-v1.5.txt
		--- vicuna-7b-v1.5_final_response.jsonl

		### [2]. Baseline
		run the comparison study. √ （factuality evaluation has not run yet）
		--eval_num=50
		the results are load in:
		/root/autodl-tmp/Results_baseline_vicuna-7b-v1.5/MASHQA=========
		--- vicuna-7b-v1.5_response.jsonl

	## ------------dataset: [MEDIQA2019]
		### [1]. MyLoop
		run the loop_new successfully. √
		[
		--eval_num=50
		--MAX_LOOP_1=3 
		--MAX_LOOP_2=2
		--...more
		]
		the results are load in:
		/root/autodl-tmp/Results_Loop_vicuna-7b-v1.5/MEDIQA2019/=========
		--- history_Loop1.txt
		--- history_Loop2.txt
		--- jduge_messages_in_Loop1_vicuna-7b-v1.5.txt
		--- vicuna-7b-v1.5_final_response.jsonl
		
		### [2]. Baseline
		run the comparison study. √
		--eval_num=50
		the results are load in:
		/root/autodl-tmp/Results_baseline_vicuna-7b-v1.5/MEDIQA2019=========
		--- vicuna-7b-v1.5_response.jsonl

		

2. # Alpaca-Lora-7b
	## ------------dataset: [MASHQA]
		### [1]. MyLoop
		run the loop_new successfully. √
		[
		--eval_num=50
		--MAX_LOOP_1=3
		--MAX_LOOP_2=2
		--...more
		]
		the results are load in: 
		/root/autodl-tmp/Results_Loop_alpaca-lora-7b/MASHQA/=========
		--- history_Loop1.txt
		--- history_Loop2.txt
		--- judge_messages_in_Loop1_Alpaca-Lora-7b.txt
		--- alpaca-lora-7b_final_response.jsonl
	
		### [2]. Baseline
		run the comparison study. √ 
		--eval_num=50
		the results are load in:
		/root/autodl-tmp/Results_baseline_alpaca-lora-7b/MASHQA=========
		--- alpaca-lora-7b_response.jsonl

	## ------------dataset: [MEDIQA2019]
		### [1]. MyLoop
		run the loop_new successfully. √
		[
		--eval_num=50
		--MAX_LOOP_1=3
		--MAX_LOOP_2=2
		--...more
		]
		the results are load in: 
		/root/autodl-tmp/Results_Loop_alpaca-lora-7b/MEDIQA/=========
		--- history_Loop1.txt
		--- history_Loop2.txt
		--- judge_messages_in_Loop1_Alpaca-Lora-7b.txt
		--- alpaca-lora-7b_final_response.jsonl
	
		### [2]. Baseline
		run the comparison study. √ 
		--eval_num=50
		the results are load in:
		/root/autodl-tmp/Results_baseline_alpaca-lora-7b/MEDIQA/=========
		--- alpaca-lora-7b_response.jsonl

3. # medAlpaca-7b
	## ------------dataset: [MASHQA]
		### [1]. MyLoop
		run the loop_new successfully. √
		[
		--eval_num=50
		--MAX_LOOP_1=3
		--MAX_LOOP_2=2
		--...more
		]
		the results are load in: 
		/root/autodl-tmp/Results_Loop_medAlpaca-7b/MASHQA/=========
		--- history_Loop1.txt
		--- history_Loop2.txt
		--- judge_messages_in_Loop1_medAlpaca-7b.txt
		--- medAlpaca-7b_final_response.jsonl
	
		### [2]. Baseline
		run the comparison study. √ 
		--eval_num=50
		the results are load in:
		/root/autodl-tmp/Results_baseline_medAlpaca-7b/MASHQA=========
		--- medAlpaca-7b_response.jsonl

	## ------------dataset: [MEDIQA2019]
		### [1]. MyLoop
		run the loop_new successfully. [not run]
		[
		--eval_num=50
		--MAX_LOOP_1=3
		--MAX_LOOP_2=2
		--...more
		]
		the results are load in: 
		/root/autodl-tmp/Results_Loop_medAlpaca-7b/MEDIQA/=========
		--- history_Loop1.txt
		--- history_Loop2.txt
		--- judge_messages_in_Loop1_medAlpaca-7b.txt
		--- medAlpaca-7b_final_response.jsonl
	
		### [2]. Baseline
		run the comparison study. √ 
		--eval_num=50
		the results are load in:
		/root/autodl-tmp/Results_baseline_medAlpaca-7b/MEDIQA/=========
		--- medAlpaca-7b_response.jsonl
=======================================================================================
* ## Evaluations: [Precision, Recall, F1, R-L, CtrlEval, MedNLI-Spl, MedNLI-Sent]
###
 	·Precision
 	·Recall
 	·F1
	·R-L
	·CtrlEval
	·MedNLI-Spl
	·MedNLI-Sent
=======================================================================================

Issue: There is much room for improvement. Some details need to be improved.
               The Loop method does not apply to every model,
               the MedAlpaca usually can't generate valid responses to a long input sentence,
               which means the Fusion-Knowledge is not applicable, and then the results of the Loop method are useless for MedAlpaca. 
