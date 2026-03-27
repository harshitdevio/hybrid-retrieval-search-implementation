from ranx import Qrels, Run, evaluate


qrels_dict = { "q_1": { "d_12": 5, "d_25": 3 },
               "q_2": { "d_11": 6, "d_22": 1 } }


run_dict = { "q_1": { "d_12": 0.9, "d_23": 0.8, "d_25": 0.7,
                      "d_36": 0.6, "d_32": 0.5, "d_35": 0.4  },
             "q_2": { "d_12": 0.9, "d_11": 0.8, "d_25": 0.7,
                      "d_36": 0.6, "d_22": 0.5, "d_35": 0.4  } }

run_dict2 = { "q_1": { "d_12": 0.9, "d_25": 0.8, "d_23": 0.8, 
                      "d_36": 0.6, "d_32": 0.5, "d_35": 0.4  },
             "q_2": { "d_11": 0.8, "d_22": 0.5, "d_25": 0.7,
                      "d_36": 0.6, "d_12": 0.8, "d_35": 0.4  } }

run_dict3 = { "q_1": { "d_12": 0.9, "d_23": 0.8, "d_25": 0.8,
                      "d_36": 0.6, "d_32": 0.5, "d_35": 0.4  },
             "q_2": { "d_11": 0.8, "d_22": 0.5, "d_25": 0.7,
                      "d_36": 0.6, "d_12": 0.8, "d_35": 0.4  } }


qrels = Qrels(qrels_dict)
run = Run(run_dict)
run2 = Run(run_dict2)
run3 = Run(run_dict3)


res = evaluate(qrels, run, "ndcg@5") # 0.7861261099276952
print(res)

res2 = evaluate(qrels, run2, "ndcg@5") # 0.981595571404941
print(res2)

res3 = evaluate(qrels, run3, "ndcg@5") # 0.9531028055697018
print(res3)