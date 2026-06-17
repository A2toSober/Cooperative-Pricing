import pygad
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from model1 import SUE_Simulator_Fast, LargeScaleGABiLevel

      
SUE_CONFIG = {
    'VOT': 0.25,
    'B_path': 10,
    'B_MODE': 5.7,
    'B_NEST': 2.85,
    'T_WAIT_PT': 8.0,
    'F_FARE_PT': 2,
    'V_BUS': 10.0,
    'A_BASE': 10.0,       
    'F_MILEAGE': 2.5,        
    'F_FUEL': 0.8,
    'P_PARK': 8.0,
    'BPR_ALPHA': 0.15,
    'BPR_BETA': 4.0,
    'loyalty_car': 0,
    'loyalty_pt': 0.1,
    'loyalty_rh': 0
}


def export_eval(sim, eval_dict, out_dir, tag):
    od_rows = []
    for (o, d), r in eval_dict["od_result"].items():
        od_rows.append({"origin_id": o, "dest_id": d, **r})
    pd.DataFrame(od_rows).to_csv(os.path.join(out_dir, f"{tag}_od_results.csv"), index=False)
    df_link = pd.DataFrame([{"car_cid": cid, "flow_total": f}
                            for cid, f in eval_dict["link_flows"].items()])
    df_link.to_csv(os.path.join(out_dir, f"{tag}_link_flows.csv"), index=False)


def main():
          
    data_dir = r"C:\Users\ASUS\Desktop\experiment3\daima\pythonProject1\xietongdingjia\model6"
    out_dir = os.getcwd()

    sim = SUE_Simulator_Fast(
        road_path=os.path.join(data_dir, "道路网络.gpkg"),
        od_path=os.path.join(data_dir, "OD_demand.csv"),
        car_path_results=os.path.join(data_dir, "car_ksp_ps_results.csv"),
        bus_path_results=os.path.join(data_dir, "bus_od2od_shortest.csv"),
        config=SUE_CONFIG
    )

                                                            
                              
                                                            
    print("\n[STEP 1] 正在计算 Baseline 场景数据...")
    n_od = len(sim.od_demand)
    baseline_eval = sim.run_sra(np.zeros(n_od), np.zeros(n_od), sim.get_init_context(),
                                max_iter=100, warm_start=False, return_detail=True)

    print("\n" + "=" * 50)
    print("BASELINE 场景指标 (全0补贴):")
    print(f"利润 (Profit): {baseline_eval['profit']:.2f}")
    print("=" * 50 + "\n")

                      
    export_eval(sim, baseline_eval, out_dir, "baseline")

                   
    n_groups = 500
    solver = LargeScaleGABiLevel(sim, n_groups=n_groups, baseline_eval=baseline_eval)

                                                            
                                   
                                                            
    group_labels = solver.group_labels
    od_keys = list(sim.od_demand.keys())

                     
    group_stats = defaultdict(lambda: {"iv_rh": [], "iv_car": [], "gc_bus": [], "rh_price": []})
    for idx, od in enumerate(od_keys):
        g = group_labels[idx]
        res = baseline_eval['od_result'][od]
        group_stats[g]["iv_rh"].append(res['rh_gc'])
        group_stats[g]["iv_car"].append(res['car_gc'])
        group_stats[g]["gc_bus"].append(res['bus_gc'])
        group_stats[g]["rh_price"].append(res['rh_money'])

    seed_heuristic_s1 = np.zeros(2 * n_groups)

                     
    for g in range(n_groups):
        v_rh = np.mean(group_stats[g]["iv_rh"])
        v_car = np.mean(group_stats[g]["iv_car"])
        v_bus = np.mean(group_stats[g]["gc_bus"])
        p_rh = np.mean(group_stats[g]["rh_price"])

                                
        if v_rh > v_car and v_rh > v_bus:
            i_rh, i_pt = (p_rh / 2.0) / 7.0, 0.0
        elif v_car >= v_rh > v_bus:
            i_rh, i_pt = (-0.5, 0.5) if (v_car - v_bus) < 5 else (-0.4, 0.4)
        elif v_rh < v_car and v_rh < v_bus:
            diff = v_bus - v_rh
            if diff > 10:
                i_rh, i_pt = 1.5, -1
            elif 2.5 < diff <= 10:
                i_rh, i_pt = 0.5, -0.5
            else:
                i_rh, i_pt = 0.25, -0.25
        else:
            i_rh, i_pt = 0.25, -0.2

        seed_heuristic_s1[2 * g] = i_rh
        seed_heuristic_s1[2 * g + 1] = i_pt

                                  
    initial_pop_s1 = np.zeros((100, 2 * n_groups))
    initial_pop_s1[0, :] = seed_heuristic_s1        
    for i in range(1, 100):
                                        
        initial_pop_s1[i, :] = seed_heuristic_s1 + np.random.uniform(-0.05, 0.05, 2 * n_groups)

                                     
                      
    gene_space_s1 = []
    for val in seed_heuristic_s1:
        gene_space_s1.append({'low': val - 0.1, 'high': val + 0.1})                       

    print(f"[STEP 2] 第一阶段启动 (策略种子微调模式)...")
    ga_s1 = pygad.GA(
        num_generations=80,                  
        sol_per_pop=100,
        initial_population=initial_pop_s1,
        num_parents_mating=30,
        fitness_func=solver.fitness_func,
        on_generation=solver.on_generation,
        gene_space=gene_space_s1,        
        mutation_probability=0.03,                  
        parallel_processing=["process", 6]
    )
    ga_s1.run()

                                                            
                                             
                                                            
    best_sol_s1, _, _ = ga_s1.best_solution()
    i_rh_ga, i_pt_ga = solver.map_to_full(best_sol_s1)

                          
    seed_s2_full = np.zeros(2 * n_od)
    for i in range(n_od):
        seed_s2_full[2 * i] = i_rh_ga[i]
        seed_s2_full[2 * i + 1] = i_pt_ga[i]

                  
    initial_pop_s2 = np.zeros((50, 2 * n_od))
    initial_pop_s2[0, :] = seed_s2_full
    for i in range(1, 50):
        initial_pop_s2[i, :] = seed_s2_full + np.random.uniform(-0.05, 0.05, 2 * n_od)

                                
    gene_space_s2 = []
    for val in seed_s2_full:
        gene_space_s2.append({'low': val - 0.05, 'high': val + 0.05})

    print(f"\n[STEP 3] 第二阶段启动 (全OD精细微调模式)...")
    ga_s2 = pygad.GA(
        num_generations=200,
        sol_per_pop=50,
        initial_population=initial_pop_s2,
        num_parents_mating=15,
        fitness_func=solver.fitness_func,
        on_generation=solver.on_generation,
        gene_space=gene_space_s2,
        mutation_probability=0.02,
        parallel_processing=["process", 6]
    )
    ga_s2.run()

                                                            
                       
                                                            
    print("\n[STEP 4] 运行最终评估并保存结果文件...")
    final_sol, _, _ = ga_s2.best_solution()
    final_rh, final_pt = final_sol[0::2], final_sol[1::2]

                        
    final_eval = sim.run_sra(final_rh, final_pt, solver.best_flow_context,
                             max_iter=100, warm_start=True, return_detail=True)

                  
    for idx, (o, d) in enumerate(od_keys):
        if (o, d) in final_eval["od_result"]:
            final_eval["od_result"][(o, d)]["I_rh_final"] = float(final_rh[idx])
            final_eval["od_result"][(o, d)]["I_pt_final"] = float(final_pt[idx])

               
    export_eval(sim, final_eval, out_dir, "final")

              
    if hasattr(solver, 'history') and len(solver.history) > 0:
        log_df = pd.DataFrame(solver.history)
        log_path = os.path.join(out_dir, "optimization_log.csv")
        log_df.to_csv(log_path, index=False)
        print(f"优化历史日志已保存至: {log_path}")

              
    # 打印对比结果汇总
    print("\n" + "=" * 55)
    print("最终方案对比汇总:")
    print(f"{'指标':<15} | {'Baseline':<12} | {'Optimized':<12} | {'提升'}")
    print(f"{'-' * 55}")
    p_diff = final_eval['profit'] - baseline_eval['profit']
    s_diff = final_eval['pt_share'] - baseline_eval['pt_share']
    print(f"{'利润':<15} | {baseline_eval['profit']:>12.2f} | {final_eval['profit']:>12.2f} | {p_diff:>+12.2f}")
    print(
        f"{'公交分担率':<15} | {baseline_eval['pt_share']:>12.2%} | {final_eval['pt_share']:>12.2%} | {s_diff:>+12.2%}")
    print(f"所有结果文件已生成于: {out_dir}")

if __name__ == "__main__":
    main()
