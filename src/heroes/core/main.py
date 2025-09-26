# heroes_optimizer.py
# Требуется: pip install psycopg2-binary pulp

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Set
from collections import defaultdict
from pathlib import Path
import os
import psycopg2
import pulp  # CBC MILP solver


def _load_env_file(path: Optional[Path] = None) -> None:
    """Load variables from a simple ``.env`` file into ``os.environ``.

    Values that are already defined in the environment are left untouched.
    """

    env_path = path or Path(__file__).resolve().parents[3] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def print_human_report(
    repo: "PgRepository",
    culture_code: str,
    cycle_code: str,
    target_building_type_code: str,
    target_level: int,
    target_count: int,
    result: OptimizationResult
):
    # 1) Собираем общий план с учетом целевого здания
    counts: Dict[Tuple[str, int], int] = defaultdict(int)
    counts[(target_building_type_code, target_level)] += target_count
    for bt, levels in result.plan.items():
        for lvl, cnt in levels.items():
            counts[(bt, lvl)] += cnt

    # 2) Тянем потоки для нужной культуры/цикла (вытянем все и отфильтруем по counts)
    with repo._conn() as conn, conn.cursor() as cur:
        cur.execute("""
            select bt.code, b.level, r.code, f.quantity
            from heroes.flow f
            join heroes.building b on b.id = f.building_id
            join heroes.building_type bt on bt.id = b.type
            join heroes.cycle cy on cy.id = f.cycle_id
            join heroes.resource r on r.id = f.resource_id
            join heroes.culture cu on cu.id = bt.culture_id
            where cu.code=%s and cy.code=%s
        """, (culture_code, cycle_code))
        rows = cur.fetchall()

    # 3) Индекс потоков по (тип,уровень)
    flows: Dict[Tuple[str, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    resources: Set[str] = set()
    for bt_code, lvl, res_code, qty in rows:
        flows[(bt_code, lvl)][res_code] += float(qty)
        resources.add(res_code)

    # 4) Агрегируем производство/потребление по ресурсам
    total_prod: Dict[str, float] = defaultdict(float)
    total_cons: Dict[str, float] = defaultdict(float)

    for (bt, lvl), cnt in counts.items():
        if cnt <= 0: continue
        for res, q in flows[(bt, lvl)].items():
            if q > 0:
                total_prod[res] += q * cnt
            elif q < 0:
                total_cons[res] += (-q) * cnt  # положительное потребление

    # 5) Красивый вывод
    def fmt_rows(rows: Iterable[Tuple[str, ...]]) -> str:
        # простой моноширинный табличный принтер
        cols = list(zip(*rows))
        widths = [max(len(x) for x in col) for col in cols]
        out = []
        for r in rows:
            out.append("  ".join(s.ljust(w) for s, w in zip(r, widths)))
        return "\n".join(out)

    print("="*72)
    print(f"Цель: {target_count}× {target_building_type_code} L{target_level}  |  Культура={culture_code}  Цикл={cycle_code}")
    print("-"*72)
    print("Нижние здания:")
    b_rows = [("Здание (ур.)", "Кол-во")]
    for (bt, lvl), cnt in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1])):
        # не печатаем целевое в этом блоке
        if bt == target_building_type_code and lvl == target_level:
            continue
        b_rows.append((f"{bt} L{lvl}", str(cnt)))
    print(fmt_rows(b_rows))
    print(f"Итого нижних зданий: {sum(int(r[1]) for r in b_rows[1:])}")
    print("-"*72)
    print("Баланс ресурсов за цикл:")
    r_rows = [("Ресурс", "Потребление", "Производство", "Профицит")]
    for res in sorted(resources):
        cons = total_cons.get(res, 0.0)
        prod = total_prod.get(res, 0.0)
        surplus = prod - cons
        if abs(cons) < 1e-9 and abs(prod) < 1e-9:
            continue
        r_rows.append((res, f"{cons:.0f}", f"{prod:.0f}", f"{surplus:.0f}"))
    print(fmt_rows(r_rows))
    print("="*72)


@dataclass(frozen=True)
class FlowRecord:
    building_id: int
    building_type_code: str
    level: int
    resource_code: str
    quantity: float  # за один производственный цикл

@dataclass
class ModelData:
    resource_codes: List[str]
    building_types: List[str]
    flows_by_building: Dict[int, List[FlowRecord]]
    building_id_by_type_level: Dict[Tuple[str, int], int]

@dataclass
class OptimizationResult:
    plan: Dict[str, Dict[int, int]]      # building_type_code -> {level: count}
    resource_surplus: Dict[str, float]   # профицит ресурсов за цикл
    total_aux_buildings: int
    status: str

class PgRepository:
    def __init__(self, dsn: Optional[str] = None):
        _load_env_file()

        if dsn:
            self.dsn = dsn
            return

        env_dsn = os.getenv("DATABASE_URL")
        if env_dsn:
            self.dsn = env_dsn
            return

        dbname = os.getenv("DB_NAME", "postgres")
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")

        dsn_parts = [
            f"dbname={dbname}",
            f"user={user}",
            f"host={host}",
        ]
        if port:
            dsn_parts.append(f"port={port}")
        if password:
            dsn_parts.append(f"password={password}")

        self.dsn = " ".join(dsn_parts)
    def _conn(self):
        return psycopg2.connect(self.dsn)

    def load_model(self, culture_code: str, cycle_code: str,
                   max_level_by_type: Dict[str, int]) -> ModelData:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute('select code from heroes.resource order by code;')
            resource_codes = [r[0] for r in cur.fetchall()]
            cur.execute('select code from heroes.building_type order by code;')
            building_types = [r[0] for r in cur.fetchall()]

            cur.execute('select id from heroes.culture where code=%s;', (culture_code,))
            culture_id = (cur.fetchone() or [None])[0]
            if culture_id is None:
                raise ValueError(f'Unknown culture {culture_code}')

            cur.execute('select id from heroes.cycle where code=%s;', (cycle_code,))
            cycle_id = (cur.fetchone() or [None])[0]
            if cycle_id is None:
                raise ValueError(f'Unknown cycle {cycle_code}')

            cur.execute('''
                select b.id, bt.code, b.level
                from heroes.building b
                join heroes.building_type bt on bt.id = b.type
                where bt.culture_id = %s
                order by bt.code, b.level
            ''', (culture_id,))
            rows = cur.fetchall()

            building_id_by_type_level = {}
            allowed_buildings = []
            for b_id, bt_code, level in rows:
                max_lvl = max_level_by_type.get(bt_code, 10**9)
                if max_lvl > 0 and level <= max_lvl:
                    building_id_by_type_level[(bt_code, level)] = b_id
                    allowed_buildings.append(b_id)
            if not allowed_buildings:
                raise ValueError('Нет доступных уровней зданий при заданных ограничениях.')

            flows_by_building = defaultdict(list)
            cur.execute('''
                select f.building_id, bt.code, b.level, r.code, f.quantity
                from heroes.flow f
                join heroes.building b on b.id = f.building_id
                join heroes.building_type bt on bt.id = b.type
                join heroes.resource r on r.id = f.resource_id
                where f.building_id = any(%s) and f.cycle_id = %s
            ''', (allowed_buildings, cycle_id))
            for b_id, bt_code, level, res_code, qty in cur.fetchall():
                flows_by_building[b_id].append(FlowRecord(b_id, bt_code, level, res_code, float(qty)))

            return ModelData(resource_codes, building_types, flows_by_building, building_id_by_type_level)

class ProductionOptimizer:
    BIG_M = 10**6
    def __init__(self, repo: PgRepository):
        self.repo = repo

    @staticmethod
    def _consumed_resources(flows_by_building: Dict[int, List[FlowRecord]]) -> List[str]:
        s = set()
        for recs in flows_by_building.values():
            for fr in recs:
                if fr.quantity < 0: s.add(fr.resource_code)
        return sorted(s)

    def optimize_for_target_building(self, culture_code: str, cycle_code: str,
                                     target_building_type_code: str, target_level: int, target_count: int,
                                     max_level_by_type: Dict[str, int],
                                     extra_limit_total_buildings: Optional[int] = None) -> OptimizationResult:
        model = self.repo.load_model(culture_code, cycle_code, max_level_by_type)

        target_building_id = model.building_id_by_type_level.get((target_building_type_code, target_level))
        if target_building_id is None:
            with self.repo._conn() as conn, conn.cursor() as cur:
                cur.execute('''
                    select b.id
                    from heroes.building b
                    join heroes.building_type bt on bt.id = b.type
                    join heroes.culture cu on cu.id = bt.culture_id
                    where bt.code=%s and b.level=%s and cu.code=%s
                ''', (target_building_type_code, target_level, culture_code))
                r = cur.fetchone()
                if not r: raise ValueError('Не найдено целевое здание.')
                target_building_id = r[0]
                cur.execute('''
                    select bt.code, r.code, f.quantity
                    from heroes.flow f
                    join heroes.building b on b.id = f.building_id
                    join heroes.building_type bt on bt.id = b.type
                    join heroes.resource r on r.id = f.resource_id
                    join heroes.cycle cy on cy.id = f.cycle_id
                    where f.building_id=%s and cy.code=%s
                ''', (target_building_id, cycle_code))
                recs = cur.fetchall()
                if not recs: raise ValueError('Для целевого здания нет потоков на заданном цикле.')
                model.flows_by_building[target_building_id] = [
                    FlowRecord(target_building_id, target_building_type_code, target_level, res, float(qty))
                    for _, res, qty in recs
                ]

        fixed_flows = defaultdict(float)
        for fr in model.flows_by_building[target_building_id]:
            fixed_flows[fr.resource_code] += fr.quantity * target_count

        decision_buildings = [bid for bid in model.flows_by_building.keys() if bid != target_building_id]

        consumed = set(self._consumed_resources(model.flows_by_building))
        consumed.update([res for res, q in fixed_flows.items() if q < 0])

        producers = defaultdict(int)
        for bid in decision_buildings:
            for fr in model.flows_by_building[bid]:
                if fr.quantity > 0: producers[fr.resource_code] += 1
        missing = [res for res in consumed if producers.get(res, 0) == 0 and fixed_flows.get(res,0) < 0]
        if missing: raise RuntimeError(f'Недостижимо: нет производителей для {", ".join(missing)}')

        prob = pulp.LpProblem('Heroes_Production_Balance', pulp.LpMinimize)
        x = {bid: pulp.LpVariable(f'x_b{bid}', lowBound=0, cat=pulp.LpInteger) for bid in decision_buildings}
        s = {res: pulp.LpVariable(f'surplus_{res}', lowBound=0, cat=pulp.LpContinuous) for res in consumed}
        prob += (self.BIG_M * pulp.lpSum(x.values())) + pulp.lpSum(s.values())

        flow_by_b_res = defaultdict(lambda: defaultdict(float))
        for bid, recs in model.flows_by_building.items():
            for fr in recs:
                flow_by_b_res[bid][fr.resource_code] += fr.quantity

        for res in consumed:
            lhs = []
            for bid in decision_buildings:
                q = flow_by_b_res[bid].get(res, 0.0)
                if q != 0.0: lhs.append(q * x[bid])
            prob += (pulp.lpSum(lhs) + fixed_flows.get(res, 0.0) - s[res] == 0), f'balance_{res}'

        if extra_limit_total_buildings is not None:
            prob += (pulp.lpSum(x.values()) <= extra_limit_total_buildings), 'limit_total_aux'

        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != 'Optimal':
            raise RuntimeError(f'Solver status: {pulp.LpStatus[status]}')

        plan = defaultdict(lambda: defaultdict(int))
        for bid, var in x.items():
            val = int(round(var.value()))
            if val <= 0: continue
            fr0 = model.flows_by_building[bid][0]
            plan[fr0.building_type_code][fr0.level] += val

        surplus = {res: float(var.value()) for res, var in s.items()}
        total_aux = sum(sum(levels.values()) for levels in plan.values())
        return OptimizationResult(plan=dict(plan), resource_surplus=surplus,
                                  total_aux_buildings=total_aux, status='Optimal')

    def optimize_for_target_resource(self, culture_code: str, cycle_code: str,
                                     target_resource_code: str,
                                     producer_building_type_code: str, producer_level: int, producer_count: int,
                                     max_level_by_type: Dict[str, int],
                                     extra_limit_total_buildings: Optional[int] = None) -> OptimizationResult:
        tmp = dict(max_level_by_type)
        tmp[producer_building_type_code] = max(tmp.get(producer_building_type_code, 0), producer_level)
        model = self.repo.load_model(culture_code, cycle_code, tmp)
        b_id = model.building_id_by_type_level.get((producer_building_type_code, producer_level))
        if b_id is None: raise ValueError('Не найден building для указанного типа/уровня.')
        produces = any(fr.resource_code == target_resource_code and fr.quantity > 0
                       for fr in model.flows_by_building[b_id])
        if not produces:
            raise ValueError(f'Здание {producer_building_type_code} L{producer_level} не производит {target_resource_code}.')
        return self.optimize_for_target_building(
            culture_code, cycle_code, producer_building_type_code, producer_level, producer_count,
            max_level_by_type, extra_limit_total_buildings
        )

if __name__ == '__main__':
    repo = PgRepository()
    opt = ProductionOptimizer(repo)
    max_lvls = {'MOTH_GLADE':3, 'RICE_FARM':3, 'THREAD_PROCESSOR':2, 'SILK_WORKSHOP':3}

    result = opt.optimize_for_target_building(
        culture_code="CHINA",
        cycle_code="H1",
        target_building_type_code="SILK_WORKSHOP",
        target_level=3,
        target_count=1,
        max_level_by_type=max_lvls
    )

    print_human_report(
        repo,
        culture_code="CHINA",
        cycle_code="H1",
        target_building_type_code="SILK_WORKSHOP",
        target_level=3,
        target_count=1,
        result=result
    )