# Семинары по машинному обучению для бакалавров 3 курса кафедры ММП и магистров 1 курса кафедр ИИТ и МФ факультета ВМК МГУ, весенний семестр 2023/2024
В репозитории находятся материалы и домашние задания по семинарам "ММРО 2022/2023"

<p align="center">
<img src="http://funzoo.ru/uploads/posts/2009-11/1258648863_tn.jpg" height=200pt> <img src="https://github.com/mmp-mmro-team/mmp_mmro_fall_2021/blob/main/trash/kernel_trick.jpg" height=200pt>
</p>

:white_check_mark: **Курс сдается через систему [anytask](https://anytask.org/course/1095). Инвайт можете получить у преподавателя**

:white_check_mark: На семинары и работу ассистентов можно [оставить отзыв](https://docs.google.com/forms/d/e/1FAIpQLSeCww7kQZRBbPDFW_dTRpKdBl1pL0jx4nezhciAof8b22O05Q/viewform)

:white_check_mark: **Полезные ссылки:**

* Текущие связанные курсы
    * [Курс Практикума, весна](https://github.com/mmp-practicum-team/mmp_practicum_spring_2024) 
    * [Общий курс ML](https://github.com/MSU-ML-COURSE/ML-COURSE-23-24)

* Осенний семестр
    * [Курс ММРО 23/24, осень](https://github.com/mmp-mmro-team/mmp_mmro_fall_2023)
    * [Курс Практикума на ЭВМ, осень](https://github.com/mmp-practicum-team/mmp_practicum_fall_2023)

* Материалы прошлых лет:
  * [Материалы древних времен](https://github.com/esokolov/ml-course-msu)
  * [Вышкинский аналог курса](https://github.com/esokolov/ml-course-hse)
  * [Курс лекций по ММРО 20/21 (в те времена он читался эксклюзивно для ММП)](http://www.machinelearning.ru/wiki/index.php?title=%D0%9C%D0%B0%D1%82%D0%B5%D0%BC%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5_%D0%BC%D0%B5%D1%82%D0%BE%D0%B4%D1%8B_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F_%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%BE%D0%B2_%28%D0%BA%D1%83%D1%80%D1%81_%D0%BB%D0%B5%D0%BA%D1%86%D0%B8%D0%B9%2C_%D0%92.%D0%92.%D0%9A%D0%B8%D1%82%D0%BE%D0%B2%29)
  * [Общий курс по МЛ 21/22](https://github.com/MSU-ML-COURSE/ML-COURSE-21-22)
  * [Курс ММРО 20/21, весна](https://github.com/mmp-mmro-team/mmp_mmro_spring_2021)
  * [Курс ММРО 21/22, весна](https://github.com/mmp-mmro-team/mmp_mmro_spring_2022)
  * [Курс ММРО 22/23, весна](https://github.com/mmp-mmro-team/-mmp_mmro_spring_2023)

# Правила выставления оценок

:white_check_mark: **По этому курсу (ММРО) в конце семестра будет экзамен**

Общая оценка по нему выставляется по следующей формуле:
![](https://github.com/mmp-mmro-team/mmp_mmro_fall_2021/blob/main/trash/formula.png)
, где 

* Check — 5 * <сумма баллов за проверочные> / <суммарный макс балл за проверочные>
* Labs — min(5, 5 * <сумма баллов за лабораторные + соревнование> / <суммарный макс балл за (лабораторные + соревнование) (без бонусов)>
* Exam — оценка за экзамен, до 5 баллов

:white_check_mark: **Обратите внимание, что проверочные проходят в рамках общекурсовых лекций по ML (называются тесты)**

Причем
* Для общей оценки 5 необходимо сдать **все (4)** _лабораторные работы_ на оценку (без учета штрафа) >= floor(1/3 * (макс. балл за работу без учета бонусов)) **и** получить за эказамен не меньше 4;
* Для общей оценки 4 необходимо сдать **не менее 3-х** работ из _всего множества лабораторных работ_ на оценку (без учета штрафа) >= floor(1/3 * (макс. балл за работу без учета бонусов)) **и** получить за экзамен не меньше 3;
* Для общей оценки 3 необходимо сдать **не менее 2-x** работ из _всего множества лабораторных работ_ на оценку (без учета штрафа) >= floor(1/3 * (макс. балл за работу без учета бонусов)) **и** получить за экзамен не меньше 3;
* floor — округление дробного числа до ближайшего целого вниз.

**Обратите внимание,** что округление общей оценки (и только ее) производится вверх.

## Оценивание контеста

* Преодоление бейзлайна (Benchmark (MMRO)) на привате --- 7 баллов (основной балл). Бенчмарк скоро появится
* Распределение 8 **бонусных** баллов на рейтингу по приватной части -- аналогично оцениванию, описанному на странице с контестом (только вместо 30 -- 8 баллов, остается разделение на 10 групп)
* Последний семинар в этом семестре будет посящен контесту. От студентов, попаших в первую (топовую) группу, мы ожидаем короткое (10 минут) выступление с презентацией по вашему решению. Без него вы сможете получить только максимум 7 бонусных баллов (вместо 8). Выступления от остальных -- по желанию (за дополнительные бонусные 0.5 балла) (при наличии большого числа желающих возможна предварительная запись на выступление). Нам будет интересно вам послушать! :)
* Итого, у контеста -- 7 основных баллов, до 8 бонусных

## Связь с общекурсовыми лекциями по ML

:white_check_mark: **По курсу лекций в конце семестра будет экзамен с оценкой**

_Если ММРО у вас обязательный курс, то:_

:white_check_mark: **Ваша оценка по общекурсовым лекциям по ML = итоговой оценке по курсу ММРО**

_В иных случах эти два курса сдаются каждый по своей системе оценивания_

# Занятия

| Дата | Номер | Тема | Материалы | ДЗ |
| :---: | :---: | --- | --- | --- |
| 8 февраля  | Семинар 1 | Кластеризация: введение, основные методы | [Кластеризация. Введение](https://github.com/mmp-mmro-team/mmp_mmro_spring_2024/blob/main/Seminars/Seminar_1__Clasterization.pdf) | ¯\\\_(ツ)\_/¯ | 
| 15 февраля | Семинар 2 | Кластеризация: спектральная кластеризация, оценки качества | [Семинар](https://github.com/mmp-mmro-team/mmp_mmro_spring_2021/blob/main/seminars/lecture17-clusterization.pdf)  | ¯\\\_(ツ)\_/¯ |
| 22 февраля | Семинар 3 | Кластеризация: обучение метрик | [Семинар](https://github.com/mmp-mmro-team/mmp_mmro_spring_2021/blob/main/seminars/sem20-knn.pdf) | [Домашнее задание на кластеризацию](https://github.com/mmp-mmro-team/mmp_mmro_spring_2024/blob/main/HomeworkN1/homework-practice-1.ipynb) |
| 29 февраля | Семинар 4 | Кластеризация: тематическое моделирование |  | ¯\\\_(ツ)\_/¯ |
| 7 марта | Семинар 5 | Оптимизация: Проксимальный метод, ADMM | [Семинар](https://github.com/mmp-mmro-team/mmp_mmro_spring_2024/blob/main/Seminars/Seminar_5__Prox_ADMM.pdf) | Ориентировочно, будет выдано соревнование до конца мая. |
| 14 марта | Семинар 6 |  Оптимизация: ускорение, стохастика, редукция дисперсии, cvxopt, jax | [Конспект](Seminars/Seminar_6__Nesterov_SGD_CVXPY/Seminar_6__Nesterov_SGD.pdf) | [Домашнее задание на оптимизацию](https://github.com/mmp-mmro-team/mmp_mmro_spring_2024/blob/main/HomeworkN2/MMF_opt_hw.ipynb) |
| 21 марта | Семинар 7 | Оптимизация: Метод Ньютона. Квазиньютоновские методы | | ¯\\\_(ツ)\_/¯  | 
| 28 марта | Семинар 8 | ЕМ-алгоритм: сходимость, скорость сходимости, связь с градиентным подъёмом |  | Домашнее задание на ЕМ-алгоритм |
| 4 апреля | Семинар 9 | ЕМ-алгоритм: решение задач (смеси распределений, GLAD) |  | ¯\\\_(ツ)\_/¯ |
| 11 апреля | Семинар 10 | OCRL(Optimat Control Reinforcement Learning): Постановка задачи, введение в ПМП, связь с ККТ |  | ¯\\\_(ツ)\_/¯ |
| 18 апреля | Семинар 11 | OCRL: функция Беллмана, динамическое программирование, связь с RL  |  | ¯\\\_(ツ)\_/¯ |
| 25 апреля | Семинар 12 | OCRL: связь  функции Беллмана и ПМП, обучение нейросетей с помощью ПМП |  | Домашнее задание на OCRL |
| 2 мая | Семинар 13 | Семинар в разработке |  | ¯\\\_(ツ)\_/¯ |
| 9 мая | Семинар 14 | Выходной  |  | ¯\\\_(ツ)\_/¯ |
| 16 мая | Семинар 15 | Разбор контеста  |  | ¯\\\_(ツ)\_/¯ |

