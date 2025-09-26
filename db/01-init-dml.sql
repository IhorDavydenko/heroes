-- heroes_init_china.sql
-- DML инициализация данных под обновлённую модель:
--   culture/cycle/resource/building_type/building/flow
-- Локация теперь называется culture (пример: CHINA).
-- === Базовые справочники ===
insert into heroes.culture(code, name) values ('CHINA', 'China') on conflict (code) do update set name = excluded.name;

insert into heroes.cycle(code, minutes)
values ('MIN30', 30), ('H1', 60), ('H2', 120)
    on conflict (code) do update set minutes = excluded.minutes;

insert into heroes.resource(code, name)
values ('MOTH_COCOON', 'Moth cocoon'), ('RICE', 'Rice'), ('THREAD', 'Thread'), ('SILK', 'Silk')
    on conflict (code) do update set name = excluded.name;
-- === Типы зданий для культуры CHINA ===
  with c as (select id from heroes.culture where code = 'CHINA')
insert
  into heroes.building_type(code, name, culture_id)
values ('MOTH_GLADE', 'Moth Glade', (select id from c)),
       ('RICE_FARM', 'Rice Farm', (select id from c)),
       ('THREAD_PROCESSOR', 'Thread Processor', (select id from c)),
       ('SILK_WORKSHOP', 'Silk Workshop', (select id from c))
    on conflict (code) do update set name = excluded.name, culture_id = excluded.culture_id;
-- === Конкретные уровни зданий ===
  with bt  as (select id, code from heroes.building_type),
       ins
           as ( insert into heroes.building (type, level) select (select id from bt where code = 'MOTH_GLADE'), 3 on conflict (type, level) do nothing returning id)
select *
  from ins;
  
  with bt  as (select id, code from heroes.building_type),
       ins
           as ( insert into heroes.building (type, level) select (select id from bt where code = 'RICE_FARM'), 2 on conflict (type, level) do nothing returning id)
select *
  from ins;
  
  with bt  as (select id, code from heroes.building_type),
       ins
           as ( insert into heroes.building (type, level) select (select id from bt where code = 'THREAD_PROCESSOR'), 2 on conflict (type, level) do nothing returning id)
select *
  from ins;
  
  with bt  as (select id, code from heroes.building_type),
       ins
           as ( insert into heroes.building (type, level) select (select id from bt where code = 'SILK_WORKSHOP'), 3 on conflict (type, level) do nothing returning id)
select *
  from ins;

-- === Потоки (cycle = H1, 60 минут) ===
-- Положительное значение -> производит; отрицательное -> потребляет.
-- Moth Glade L3: +1033 MOTH_COCOON
insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, 1033.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'MOTH_GLADE'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'MOTH_COCOON'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 3
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;
-- Rice Farm L2: +375 RICE
insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, 375.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'RICE_FARM'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'RICE'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 2
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;
-- Thread Processor L2: +200 THREAD, -270 RICE, -910 MOTH_COCOON
insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, 200.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'THREAD_PROCESSOR'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'THREAD'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 2
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;

insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, -270.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'THREAD_PROCESSOR'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'RICE'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 2
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;

insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, -910.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'THREAD_PROCESSOR'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'MOTH_COCOON'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 2
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;
-- Silk Workshop L3: +270 SILK, -490 THREAD, -580 RICE
insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, 270.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'SILK_WORKSHOP'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'SILK'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 3
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;

insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, -490.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'SILK_WORKSHOP'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'THREAD'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 3
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;

insert into heroes.flow(building_id, resource_id, cycle_id, quantity)
select b.id, r.id, cy.id, -580.0
  from heroes.building b
       join heroes.building_type bt on bt.id = b.type and bt.code = 'SILK_WORKSHOP'
       join heroes.culture cu on cu.id = bt.culture_id and cu.code = 'CHINA'
       join heroes.resource r on r.code = 'RICE'
       join heroes.cycle cy on cy.code = 'H1'
 where b.level = 3
    on conflict (building_id, cycle_id, resource_id) do update set quantity = excluded.quantity;
