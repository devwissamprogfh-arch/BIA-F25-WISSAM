# Smart Academic Scheduler — الفريق الأول: المشكلة والبيانات والتقييم

مشروع الخوارزميات الذكية: جدولة المحاضرات والامتحانات باستخدام GA و PSO و ACO.
هذا الجزء (الفريق الأول) يسلّم **محرك التقييم الموثوق** الذي تستخدمه الخوارزميات الثلاث:

| الملف | الوظيفة |
|---|---|
| `models/entities.py` | الكيانات: `TimeGrid`, `Teacher`, `Room`, `Student`, `Course`, `Event` |
| `models/lecture.py`, `models/exam.py` | بناء أحداث المحاضرات والامتحانات من المقررات |
| `models/schedule.py` | تمثيل الجدول `Schedule` (قابل للاستخدام في GA/PSO/ACO) وقراءة/كتابة CSV |
| `core/problem.py` | الصنف `Problem` + تحميل البيانات + جداول بحث جاهزة للخوارزميات |
| `core/constraints.py` | فهرس القيود الصارمة والمرنة وأوزان العقوبة |
| `core/fitness.py` | دالة الهدف المشتركة `evaluate()` / `fitness()` |
| `core/validator.py` | المدقق المستقل `validate_schedule()` وفاحص البيانات `diagnose_instance()` |
| `core/baseline.py` | خطّا أساس: عشوائي وجشع (للمقارنة والفحص) |
| `data/generate_data.py` | مولّد بيانات اصطناعية بحل صالح مزروع |
| `data/describe_data.py` | تقرير جودة البيانات والنتائج المرجعية |
| `tests/` | 41 اختبارًا لكشف التعارضات وصحة البيانات |

> ملاحظة: أضفنا ملفات `models/entities.py` و`core/problem.py` و`core/baseline.py`
> و`data/*.py` إلى الهيكل المقترح في الخطة. بقية الهيكل لم يتغير.
> مجلدات `algorithms/` و`experiments/` و`visualization/` جاهزة للفرق الأخرى.

## المتطلبات

Python 3.9 أو أحدث. **لا توجد مكتبات خارجية** لهذا الجزء (انظر `requirements.txt`).

## طريقة التشغيل

```bash
# 1) (optional) create a virtual environment
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2) run all tests (41 tests, < 1 second)
python -m unittest discover -s tests -t .

# 3) regenerate the six data sets (deterministic: same seeds -> same data)
python -m data.generate_data

# 4) print the data-quality report and baseline results
python -m data.describe_data

# 5) print the baseline table only
python -m core.baseline
```

## البيانات

ست مجموعات في `data/<mode>/<size>/` حيث `mode` = `lecture` أو `exam`، و`size` = `small` (20 حدثًا)
أو `medium` (60) أو `large` (150). كل مجموعة تحوي:
`courses.csv`, `rooms.csv`, `teachers.csv`, `students.csv`, `enrollments.csv`, `meta.json`
و`planted_schedule.csv` (جدول صالح معروف مسبقًا: صفر مخالفات صارمة).

البيانات **اصطناعية** وليست بيانات جامعية حقيقية.

## القيود

| القيد | النوع | وحدة العدّ |
|---|---|---|
| `room_conflicts` | صارم | زوج أحداث في القاعة نفسها وبزمن متداخل |
| `teacher_conflicts` | صارم | زوج أحداث لأستاذ واحد وبزمن متداخل |
| `student_conflicts` | صارم | طالب × زوج متداخل (يُعدّ كل طالب متأثر) |
| `capacity_violations` | صارم | حدث تتجاوز أعداد حضوره سعة القاعة |
| `room_type_violations` | صارم | نوع القاعة لا يطابق المطلوب (مخبر/عادية) |
| `availability_violations` | صارم | حدث يتقاطع مع وقت عدم توفر الأستاذ |
| `bounds_violations` | صارم | حدث غير مُسند، أو قيمة خارج المجال، أو يتجاوز نهاية اليوم |
| `preference_violations` | مرن | حدث خارج الفترات المفضلة للأستاذ |
| `spread_violations` | مرن | جلستان لمقرر واحد في اليوم نفسه |
| `gap_violations` | مرن | طالب لديه حدثان متقاربان أكثر من اللازم في اليوم نفسه |
| `room_waste` | مرن | مجموع (السعة − الحضور)/السعة |

الأوزان الافتراضية: 1000 لكل قيد صارم، و 2 / 3 / 2 / 1 للقيود المرنة بالترتيب أعلاه
(`core/constraints.py`). يمكن تعديلها بـ `FitnessConfig().with_overrides(...)`.

## أمثلة استخدام (للفرق الأخرى)

```python
import random
from core.problem import load_problem
from core.fitness import evaluate, fitness
from core.validator import validate_schedule, diagnose_instance
from core.baseline import random_schedule
from models.schedule import Schedule

# load one data set
problem = load_problem("data/lecture/small")

# check the data itself before running any algorithm
print(diagnose_instance(problem).text())

# search-space information prepared for the algorithms
i = 0                                       # event index
starts = problem.valid_starts[i]            # start slots that fit inside a day
rooms = problem.feasible_rooms[i]           # rooms of the right type and enough seats

# a schedule = two integer lists aligned with problem.events
schedule = random_schedule(problem, random.Random(42))
schedule.assign(i, starts[0], rooms[0])     # place event i

# evaluate: scalar objective for the algorithm, details for the report
value = fitness(problem, schedule)          # lower is better
result = evaluate(problem, schedule)
print(result.hard_violations, result.soft_penalty, result.feasible)

# PSO-style flat vector: [slot0, room0, slot1, room1, ...]
vector = schedule.to_vector()
same = Schedule.from_vector(vector)

# final answer must always be confirmed by the independent validator
report = validate_schedule(problem, schedule)
print(report.text())
```

التفاصيل الكاملة للاتفاق بين الفرق في `docs/team_interface.md`.

## الخوارزميات (GA / PSO / ACO)

الخوارزميات الثلاث **هجينة (memetic)**: لكل منها آليته الأساسية، وتشترك في محرك بحث محلي
`algorithms/common.py` لإصلاح القيود الصارمة. بدون هذا الإصلاح كانت GA و PSO تنتهي بمئات الانتهاكات
الصارمة على الأحجام المتوسطة والكبيرة.

| الخوارزمية | الآلية الأساسية | دور الإصلاح المحلي |
|---|---|---|
| GA  (`genetic_algorithm.py`) | تمثيل (slot, room) لكل حدث، اختيار tournament، تزاوج منتظم، طفرة داخل النطاق الصالح، نخبوية | يُصلح الأبناء المخالفين باحتمال `repair_rate` |
| PSO (`particle_swarm.py`) | متجه حقيقي من فهارس النطاقات، تحديث السرعة القياسي، فك ترميز بالتقريب | يُصلح كل جسيم مخالف ثم يُعاد ترميز موضعه |
| ACO (`ant_colony.py`) | فيرومون على (slot, room) لكل حدث + دالة استدلال من الكلفة التزايدية، تبخر وتعزيز (حدود MAX-MIN) | يُصلح النمل المخالف قبل التقييم والتعزيز |

* `common.SearchState`: كلفة تزايدية لنقل حدث واحد (نفس أوزان `core.fitness`، والاختبار يتحقق من التطابق).
* `common.construct`: بناء جشع عشوائي (الأحداث الأصعب أولاً).
* `common.repair`: إصلاح *breakout* (min-conflicts موزون): يحرّك الأحداث المتعارضة فقط، وعند الوصول لحد أدنى
  محلي يرفع وزن الأزواج المتعارضة. بيانات التعارض تُحدَّث تزايدياً (O(الجيران) لكل خطوة).
* بعد الحلقة الرئيسية، إن بقي أفضل حل مخالفاً، تُجرى محاولة صقل أخيرة (`final_repair_steps`).
* كل خوارزمية تُعيد `best_hard_violations` الحقيقي؛ والنتيجة النهائية تُؤكَّد دائماً بـ `validate_schedule`.

اختبارات الخوارزميات: `python -m unittest discover -s tests -t .` (48 اختبارًا).

> **ملاحظة عن lecture/large:** هذه المجموعة شديدة التقييد (حوالي 63% من أزواج الأحداث تشترك بطلاب،
> والحل المزروع يكاد يكون الوحيد). الخوارزميات الثلاث تنزل من مئات الانتهاكات إلى بضع عشرات أو أقل لكن لا
> تضمن الوصول إلى الصفر؛ انظر جدول النتائج في التقرير.
