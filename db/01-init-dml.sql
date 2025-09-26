-- heroes_init_china.sql
-- DML инициализация данных под обновлённую модель:
--   culture/cycle/resource/building/flow
-- Локация теперь называется culture (пример: CHINA).
-- === Базовые справочники ===
insert into heroes.culture(code, name) values ('CHINA', 'China') on conflict (code) do update set name = excluded.name;

insert into heroes.cycle(code, name, minutes)
values ('MIN30', '30 minutes', 30), ('H1', '1 hour', 60), ('H2', '2 hours', 120)
    on conflict (code) do update set name = excluded.name, minutes = excluded.minutes;

insert into heroes.resource(code, name)
values ('MOTH_COCOON', 'Moth cocoon'), ('RICE', 'Rice'), ('THREAD', 'Thread'), ('SILK', 'Silk')
    on conflict (code) do update set name = excluded.name;
-- === Здания для культуры CHINA ===
  with c as (select id from heroes.culture where code = 'CHINA')
insert
  into heroes.building(code, name, culture_id)
values ('MOTH_GLADE', 'Moth Glade', (select id from c)),
       ('RICE_FARM', 'Rice Farm', (select id from c)),
       ('THREAD_PROCESSOR', 'Thread Processor', (select id from c)),
       ('SILK_WORKSHOP', 'Silk Workshop', (select id from c))
    on conflict (code) do update set name = excluded.name, culture_id = excluded.culture_id;

-- === Потоки (cycle = H1, 60 минут) ===
-- Положительное значение -> производит; отрицательное -> потребляет.
-- Moth Glade L3: +1033 MOTH_COCOON
insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'MOTH_GLADE', 3, 'MOTH_COCOON', 'H1', 1033.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;
-- Rice Farm L2: +375 RICE
insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'RICE_FARM', 2, 'RICE', 'H1', 375.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;
-- Thread Processor L2: +200 THREAD, -270 RICE, -910 MOTH_COCOON
insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'THREAD_PROCESSOR', 2, 'THREAD', 'H1', 200.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;

insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'THREAD_PROCESSOR', 2, 'RICE', 'H1', -270.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;

insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'THREAD_PROCESSOR', 2, 'MOTH_COCOON', 'H1', -910.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;
-- Silk Workshop L3: +270 SILK, -490 THREAD, -580 RICE
insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'SILK_WORKSHOP', 3, 'SILK', 'H1', 270.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;

insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'SILK_WORKSHOP', 3, 'THREAD', 'H1', -490.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;

insert into heroes.flow(culture, building, level, resource, cycle, quantity)
values ('CHINA', 'SILK_WORKSHOP', 3, 'RICE', 'H1', -580.0)
    on conflict (culture, building, level, resource, cycle) do update set quantity = excluded.quantity;
