select * from heroes.building;
5,KAOLIN_QUARRY
6,CLAY_PROCESSOR
7,PORCELAIN_WORKSHOP

select * from heroes.resource;
5,KAOLIN,Kaolin
6,CLAY,Clay
7,PORCELAIN,Porcelain

selecT * from heroes.cycle;
1,MIN30
2,H1
3,H2
4,MIN5

select * from heroes.flow where culture = 'CHINA' order by 1, 2, 3, 4, 5;
select * from heroes.flow where culture = 'CHINA' and building = 'CLAY_PROCESSOR' and level >2 order by 1, 2, 3, 4, 5;
select * from heroes.flow where culture = 'CHINA' and building = 'PORCELAIN_WORKSHOP' order by 1, 2, 3, 4, 5;
select * from heroes.flow where culture = 'CHINA' and building = 'PORCELAIN_WORKSHOP' order by 1, 2, 3, 4, 5;

