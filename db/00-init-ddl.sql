-- 1) Схема
create schema if not exists heroes;
-- 2) Справочники
create table if not exists heroes.culture
(
  id   serial primary key,
  code text unique not null,
  name text        not null
);

create table if not exists heroes.cycle
(
  id      serial primary key,
  code    text unique not null, -- 'MIN30' | 'H1' | 'H2'
  minutes int         not null check (minutes > 0)
);

create table if not exists heroes.resource
(
  id   serial primary key,
  code text unique not null, -- 'MOTH_COCOON', 'RICE', ...
  name text        not null
);

create table if not exists heroes.building_type
(
  id         serial primary key,
  code       text unique not null, -- 'MOTH_GLADE', 'RICE_FARM', ...
  name       text        not null,
  culture_id int         not null references heroes.culture (id) on delete cascade
);
-- 3) Конкретные уровни зданий в локации
create table if not exists heroes.building
(
  id    serial primary key,
  type  int not null references heroes.building_type (id) on delete cascade,
  level int not null check (level >= 1),
  unique (type, level)
);

-- 4) Потоки ресурсов по зданиям/уровням/интервалам.
--    quantity > 0  -> производит; quantity < 0 -> потребляет.
create table if not exists heroes.flow
(
  building_id int            not null references heroes.building (id) on delete cascade,
  resource_id int            not null references heroes.resource (id) on delete cascade,
  cycle_id    int            not null references heroes.cycle (id) on delete cascade,
  quantity    numeric(18, 6) not null, -- штук за интервал
  primary key (building_id, cycle_id, resource_id)
);
-- 5) Индексы-ускорители
create index if not exists ix_flow_res on heroes.flow (resource_id);

