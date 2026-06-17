import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SUE_Simulator_Fast:
    def __init__(self, road_path, od_path, car_path_results, bus_path_results, config):
        self.conf = config
        import geopandas as gpd

        print("正在加载基础路网与OD数据...")
        road_gdf = gpd.read_file(road_path)

        self.links = {
            str(int(r["car_cid"])): {
                "t0": float(r["TIME"]),
                "cap": float(r["CAPACITY"]),
                "len_km": float(r["LENGTH"]) / 1000.0
            } for _, r in road_gdf.iterrows()
        }
        self.link_ids = list(self.links.keys())

        df_od = pd.read_csv(od_path, index_col=0)
        self.od_demand = {}
        for o in df_od.index:
            for d in df_od.columns:
                q = float(df_od.loc[o, d])
                if q > 0:
                    self.od_demand[(int(o), int(d))] = q

        print(f"成功读取有效 OD 对数量: {len(self.od_demand)}")

        print(f"正在建立路径索引 ({len(self.od_demand)} OD)...")
        self.car_paths_dict = defaultdict(list)
        df_car = pd.read_csv(car_path_results)

        for row in df_car.itertuples():
            od = (int(row.origin_id), int(row.dest_id))
            edges = [s for s in str(row.edge_car_cids).split(";") if s in self.links]
            self.car_paths_dict[od].append({
                "edges": edges,
                "len_km": float(row.len_sum) / 1000.0,
                "overlap": float(getattr(row, "overlap_factor", 1.0))
            })

        self.bus_paths_dict = pd.read_csv(bus_path_results).set_index(["origin_id", "dest_id"]).to_dict('index')

    def get_init_context(self):
        return {
            "link_flows": {cid: 0.0 for cid in self.link_ids},
            "bus_history": {od: 0.0 for od in self.od_demand}
        }

    def run_sra(self, I_rh_vec, I_pt_vec, flow_context, max_iter=30, warm_start=True, return_detail=False):
        L_car, L_pt, L_rh = self.conf.get('loyalty_car', 0.0), self.conf.get('loyalty_pt', 0.0), self.conf.get(
            'loyalty_rh', 0.0)
        L_sum = L_car + L_pt + L_rh

        current_flows = flow_context["link_flows"].copy() if warm_start else {cid: 0.0 for cid in self.link_ids}
        bus_history = flow_context["bus_history"].copy() if warm_start else {od: 0.0 for od in self.od_demand}
        iters = max(1, int(max_iter)) if warm_start else int(max_iter)

        od_keys = list(self.od_demand.keys())
        num_od = len(od_keys)
        od_result = {} if return_detail else None
        v2_vector = np.zeros(num_od)


        total_profit, total_q_bus, total_q_all = 0.0, 0.0, 0.0
        v2_vector = np.zeros(num_od)

        for n in range(1, iters + 1):
            alpha_n = 1.0 / n
            new_flows = {cid: 0.0 for cid in self.link_ids}
            temp_bus_flows = {od: 0.0 for od in self.od_demand}
            total_profit, total_q_bus, total_q_all = 0.0, 0.0, 0.0

            for idx, (o, d) in enumerate(od_keys):
                total_q = self.od_demand[(o, d)]
                i_rh, i_pt = float(I_rh_vec[idx]), float(I_pt_vec[idx])


                bus_info = self.bus_paths_dict.get((o, d))
                if bus_info:
                    bus_dist = bus_info['total_length'] / 1000.0
                    bus_time = (self.conf['T_WAIT_PT'] + (bus_dist / self.conf['V_BUS']) * 60.0) * \
                               (1.0 + 1 * (bus_history[(o, d)] / 100.0) ** 2) + bus_dist
                    bus_money = self.conf['F_FARE_PT'] - i_pt
                    gc_bus = bus_time + bus_money / self.conf['VOT']
                else:
                    gc_bus, bus_time, bus_money, bus_dist = 1e12, 1e12, 1e12, 0.0

                paths = self.car_paths_dict.get((o, d), [])
                if not paths:
                    v2_vector[idx] = gc_bus
                    total_q_bus += total_q
                    total_q_all += total_q
                    temp_bus_flows[(o, d)] = total_q
                    if return_detail and n == iters:
                        od_result[(o, d)] = {"demand": total_q, "q_bus": total_q, "bus_gc": gc_bus,
                                             "bus_dist_km": bus_dist}
                    continue


                path_times = [sum(self._get_bpr_time(cid, current_flows[cid]) for cid in p['edges']) for p in paths]
                car_moneys = [(p['len_km'] * self.conf['F_FUEL'] + self.conf['P_PARK']) for p in paths]
                rh_moneys = [
                    (
                            self.conf['A_BASE'] +
                            max(0, p['len_km'] - 3.0) * self.conf['F_MILEAGE'] -
                            i_rh
                    ) for p in paths
                ]

                car_gc = [t + m / self.conf['VOT'] for t, m in zip(path_times, car_moneys)]
                rh_gc = [t + m / self.conf['VOT'] for t, m in zip(path_times, rh_moneys)]
                ps = [p['overlap'] for p in paths]


                p_path_car, iv_car = self._weibit(car_gc, ps, self.conf['B_path'])
                p_path_rh, iv_rh = self._weibit(rh_gc, ps, self.conf['B_path'])
                p_within, iv_priv = self._weibit([gc_bus, iv_rh], [1, 1], self.conf['B_MODE'])
                p_nest, v_total_system = self._weibit([iv_priv, iv_car], [1, 1], self.conf['B_NEST'])

                q_car = total_q * ((1 - L_sum) * (p_nest[1]) + L_car)
                q_rh = total_q * ((1 - L_sum) * (p_nest[0] * p_within[1]) + L_rh)
                q_bus = total_q * ((1 - L_sum) * (p_nest[0] * p_within[0]) + L_pt)


                total_profit += (
                        0.25 * q_rh * (self.conf['A_BASE'] + max(0, paths[0]['len_km'] - 3) * self.conf['F_MILEAGE'] - i_rh)
                        + q_bus * (self.conf['F_FARE_PT'] - i_pt)
                )
                total_q_bus, total_q_all = total_q_bus + q_bus, total_q_all + total_q
                v2_vector[idx] = v_total_system
                temp_bus_flows[(o, d)] = q_bus

                for i, p in enumerate(paths):
                    f = q_car * p_path_car[i] + q_rh * p_path_rh[i]
                    for cid in p['edges']: new_flows[cid] += f


                if return_detail and n == iters:
                    od_result[(o, d)] = {
                        "demand": float(total_q), "q_car": float(q_car), "q_rh": float(q_rh), "q_bus": float(q_bus),
                        "share_bus": float(q_bus / total_q), "v2_total_utility": float(v_total_system),
                        "car_time": float(path_times[0]), "car_money": float(car_moneys[0]), "car_gc": float(car_gc[0]),
                        "rh_time": float(path_times[0]), "rh_money": float(rh_moneys[0]), "rh_gc": float(rh_gc[0]),
                        "bus_time": float(bus_time), "bus_money": float(bus_money), "bus_gc": float(gc_bus),
                        "bus_dist_km": float(bus_dist)
                    }

            for cid in self.link_ids: current_flows[cid] += alpha_n * (new_flows[cid] - current_flows[cid])
            for od in self.od_demand: bus_history[od] += alpha_n * (temp_bus_flows[od] - bus_history[od])

        flow_context["link_flows"], flow_context["bus_history"] = current_flows, bus_history
        pt_share = (total_q_bus / total_q_all) if total_q_all > 0 else 0.0

        if not return_detail: return float(total_profit), float(pt_share), v2_vector
        return {"profit": float(total_profit), "pt_share": float(pt_share), "v2_vector": v2_vector,
                "od_result": od_result, "link_flows": current_flows}

    def _get_bpr_time(self, cid, flow):
        l = self.links[cid]
        return l["t0"] * (1.0 + self.conf['BPR_ALPHA'] * (flow / l["cap"]) ** self.conf['BPR_BETA'])

    def _weibit(self, costs, ps, beta):
        v = np.array(ps) * np.power(np.maximum(costs, 1e-6), -beta)
        sum_v = np.sum(v)
        return (v / sum_v, float(np.power(sum_v, -1.0 / beta))) if sum_v > 0 else (
        np.ones(len(costs)) / len(costs), 1e12)


class LargeScaleGABiLevel:
    def __init__(self, simulator, n_groups=50, baseline_eval=None):
        self.sim = simulator
        self.n_groups = n_groups
        self.n_od = len(self.sim.od_demand)
        self.best_flow_context = self.sim.get_init_context()
        self.history = []


        if baseline_eval is not None and "od_result" in baseline_eval:
            self.base_profit = baseline_eval['profit']
            self.base_sh = baseline_eval['pt_share']
            self.base_v2 = baseline_eval['v2_vector']
            self.baseline_detail = baseline_eval
        else:
            print("[Warning] 正在计算高精度基准数据以支持多维聚类...")
            self.baseline_detail = self.sim.run_sra(
                np.zeros(self.n_od), np.zeros(self.n_od),
                self.best_flow_context, max_iter=100, warm_start=False, return_detail=True
            )
            self.base_profit = self.baseline_detail['profit']
            self.base_sh = self.baseline_detail['pt_share']
            self.base_v2 = self.baseline_detail['v2_vector']


        print("\n" + "=" * 40)
        print("   BASELINE 初始状态确认")
        print(f"   利润 (Profit): {self.base_profit:.2f}")
        print(f"   公交分担率 (PT Share): {self.base_sh:.2%}")
        print("=" * 40 + "\n")

        self.history.append({
            "stage": "Baseline", "gen": 0, "fit": 0,
            "profit": self.base_profit, "sh": self.base_sh, "violation": 0
        })


        features_list = []
        od_keys = list(self.sim.od_demand.keys())
        for od in od_keys:
            res = self.baseline_detail['od_result'][od]
            features_list.append([
                res.get('rh_gc', 0),
                res.get('car_gc', 0),
                res.get('bus_gc', 0),
                res.get('demand', 0)
            ])


        features = StandardScaler().fit_transform(np.array(features_list))
        self.group_labels = KMeans(
            n_clusters=n_groups,
            random_state=42,
            n_init=10
        ).fit(features).labels_

        print(f"已基于成本结构特征完成 {n_groups} 个群组的聚类。")

    def map_to_full(self, solution):
        i_rh, i_pt = np.zeros(self.n_od), np.zeros(self.n_od)
        for g in range(self.n_groups):
            mask = (self.group_labels == g)
            i_rh[mask], i_pt[mask] = solution[2 * g], solution[2 * g + 1]
        return i_rh, i_pt

    def fitness_func(self, ga_instance, solution, solution_idx):

        is_stage_2 = (len(solution) == 2 * self.n_od)


        current_gen = ga_instance.generations_completed
        total_gen = ga_instance.num_generations

        if not is_stage_2:
            i_rh, i_pt = self.map_to_full(solution)
            m_iter = 20
        else:
            i_rh, i_pt = solution[0::2], solution[1::2]
            m_iter = 20


        local_context = {"link_flows": self.best_flow_context["link_flows"].copy(),
                         "bus_history": self.best_flow_context["bus_history"].copy()}
        profit, pt_share, v2_curr = self.sim.run_sra(i_rh, i_pt, local_context,
                                                     max_iter=m_iter, warm_start=True)


        v2_diff = v2_curr - self.base_v2
        v_mask = v2_diff > 0.05
        num_violated = np.sum(v_mask)
        sum_violated = np.sum(v2_diff[v_mask])


        if not is_stage_2:


            progress = current_gen / max(total_gen, 1)


            sh_penalty_weight = 1e3 * (0.3 + 0.7 * progress)
            sh_penalty = max(0, (self.base_sh - 0.01) - pt_share) * sh_penalty_weight


            num_viol_base = 10000
            sum_viol_base = 10000


            escalation = 0.3 + 1.7 * (progress ** 2)

            u_penalty = (num_violated * num_viol_base * escalation) + \
                        (sum_violated * sum_viol_base * escalation)


            efficiency_penalty = np.sum(np.abs(i_rh) + np.abs(i_pt)) * 0.001


            early_exploration_bonus = 0
            if progress < 0.1:

                if num_violated < len(self.sim.od_demand) * 0.1 and profit > self.base_profit * 0.8:
                    early_exploration_bonus = 50 * (1 - progress / 0.1)

            return profit - sh_penalty - u_penalty - efficiency_penalty + early_exploration_bonus

        else:

            sh_penalty = max(0, (self.base_sh - 0.01) - pt_share) * 5e4
            u_penalty = (num_violated * 50) + (sum_violated * 1000)
            profit_loss_penalty = max(0, self.base_profit - profit) * 200

            return profit - sh_penalty - u_penalty - profit_loss_penalty

    def on_generation(self, ga_instance):
        sol, fit, _ = ga_instance.best_solution()
        i_rh, i_pt = self.map_to_full(sol) if len(sol) == 2 * self.n_groups else (sol[0::2], sol[1::2])
        tag = "全局" if len(sol) == 2 * self.n_groups else "精细"
        p, sh, v2_curr = self.sim.run_sra(i_rh, i_pt, self.best_flow_context, max_iter=10, warm_start=True,
                                          return_detail=False)
        num_violated = np.sum((v2_curr - self.base_v2) > 0.05)
        self.history.append({"stage": tag, "gen": ga_instance.generations_completed, "fit": fit, "profit": p, "sh": sh,
                             "violation": num_violated})
        print(
            f" >>> [{tag}阶段] 代 {ga_instance.generations_completed:02d} | 适应度: {fit:.2f} | 利润: {p:.2f}")
