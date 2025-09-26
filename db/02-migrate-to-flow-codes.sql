-- Миграция со старой схемы (building_type/building/flow с ID) на схему с кодами.
begin;

-- 1) Добавляем человекочитаемое имя циклам, если его ещё нет.
alter table heroes.cycle add column if not exists name text;
update heroes.cycle set name = coalesce(name, code);
alter table heroes.cycle alter column name set not null;

-- 2) Подготавливаем новую таблицу справочника зданий (бывший building_type).
create table if not exists heroes.building_new
(
  id         serial primary key,
  code       text unique not null,
  name       text        not null,
  culture_id int         not null references heroes.culture (id) on delete cascade
);

insert into heroes.building_new(code, name, culture_id)
select bt.code, bt.name, bt.culture_id
  from heroes.building_type bt
on conflict (code) do update set name = excluded.name, culture_id = excluded.culture_id;

-- 3) Переносим потоки в новую структуру, где ключом служат коды.
create table if not exists heroes.flow_new
(
  culture  text           not null references heroes.culture (code) on delete cascade,
  building text           not null references heroes.building_new (code) on delete cascade,
  level    int            not null check (level >= 1),
  resource text           not null references heroes.resource (code) on delete cascade,
  cycle    text           not null references heroes.cycle (code) on delete cascade,
  quantity numeric(18, 6) not null,
  primary key (culture, building, level, resource, cycle)
);

insert into heroes.flow_new(culture, building, level, resource, cycle, quantity)
select cu.code,
       bt.code,
       b.level,
       r.code,
       cy.code,
       f.quantity
  from heroes.flow f
       join heroes.building b on b.id = f.building_id
       join heroes.building_type bt on bt.id = b.type
       join heroes.culture cu on cu.id = bt.culture_id
       join heroes.resource r on r.id = f.resource_id
       join heroes.cycle cy on cy.id = f.cycle_id
on conflict (culture, building, level, resource, cycle)
  do update set quantity = excluded.quantity;

-- 4) Удаляем больше не используемые таблицы и переименовываем новые.
drop table if exists heroes.flow;
drop table if exists heroes.building;
drop table if exists heroes.building_type;

drop sequence if exists heroes.building_id_seq;
drop sequence if exists heroes.building_type_id_seq;

alter table heroes.building_new rename to building;
alter sequence if exists heroes.building_new_id_seq rename to building_id_seq;

alter table heroes.flow_new rename to flow;
create index if not exists ix_flow_res on heroes.flow (resource);

commit;
