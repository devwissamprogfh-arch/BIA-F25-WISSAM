# Smart Academic Scheduler — الجدولة الذكية للمحاضرات والامتحانات (GA / PSO / ACO)

**المقرر:** الخوارزميات الذكية (BIA F25) — **مالك المشروع:** WISSAM
**المستودع:** https://github.com/devwissamprogfh-arch/BIA-F25-WISSAM.git

## 1) ما هو المشروع؟

تطبيق يبني **جدولًا أسبوعيًا** للمحاضرات (5 أيام) أو للامتحانات (10 أيام) بحيث يُعطى كل حدث (محاضرة/امتحان) **وقت بداية + قاعة**،
مع تجنّب التعارضات (قاعة، مدرّس، طلاب، سعة، نوع القاعة، عدم توفر المدرّس) وتحسين تفضيلات مرنة (توزيع متوازن، تقليل الفجوات…).
نحلّ المسألة بثلاث خوارزميات ذكية ونقارنها: **الخوارزمية الجينية GA**، **سرب الجسيمات PSO**، **مستعمرة النمل ACO**.

> الخوارزميات الثلاث **هجينة (Memetic)**: خوارزمية الاستكشاف + بحث محلي قصير `repair` لإصلاح التعارضات
> (الكود المشترك في `algorithms/common.py`). بدونه لا تصل GA/PSO/ACO النقية إلى جدول صالح على الأحجام الكبيرة.

## 2) المتطلبات والتثبيت

- Python 3.9 أو أحدث.
- المحرك والخوارزميات والاختبارات: **مكتبات بايثون القياسية فقط**.
- واجهة التطبيق `app.py` تحتاج: `pandas` و `streamlit`.

```bash
git clone https://github.com/devwissamprogfh-arch/BIA-F25-WISSAM.git
cd BIA-F25-WISSAM
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3) كيف تشغّل التطبيق (الطريقة الأساسية للدكتور)

```bash
streamlit run app.py
```

يفتح المتصفح تلقائيًا. من القائمة الجانبية اختر بالترتيب:

| الخيار | المعنى |
|---|---|
| نوع المسألة | **محاضرات** أو **امتحانات** (اختياري؛ كل نوع له بيانات وفترة زمنية مختلفة) |
| الحجم | small (20 حدثًا) / medium (60) / large (150) |
| الخوارزمية | GA أو PSO أو ACO |
| الميزانية | quick (سريع) أو standard (المعايير المستخدمة في التقرير) |
| بذرة البداية | الافتراضي 42 (نفس البذرة = نفس النتيجة) |
| عدد التكرارات للتجربة | الافتراضي 3 لقياس **استقرار** النتائج |
| قياس الذاكرة | اختياري (يبطئ التنفيذ؛ يُجرى تشغيل إضافي واحد) |

ثم اضغط **تشغيل**. تظهر خمسة أقسام:

1. **جودة الحل:** قيمة Fitness، عدد المخالفات الصارمة (الهدف 0)، العقوبات المرنة، وحالة المدقق المستقل (`valid`).
2. **زمن التنفيذ والموارد وسرعة التقارب:** زمن الجدار/المعالج، الذاكرة، التكرار الذي بلغ فيه الحل 95% من التحسن وأول تكرار خالٍ من المخالفات.
3. **منحنى التقارب:** Iteration → Best Fitness.
4. **استقرار النتائج:** المتوسط والانحراف المعياري ومعامل التباين وأفضل/أسوأ قيمة عبر عدة بذور.
5. **الجدول الأسبوعي:** الأيام × الساعات × القاعات، مع فلتر للقاعة وقائمة تفصيلية بالأحداث. عدم وجود تعارض يُتحقق منه بالمدقق المستقل: يجب أن تكون «المخالفات الصارمة = 0» و`valid = True`.

أزرار التنزيل تصدّر النتائج (summary / runs / convergence / schedule) بصيغة CSV.

### كيف أعرف أن الجدول بلا تعارض؟
اقرأ بطاقة «جودة الحل»: إن كانت **المخالفات الصارمة = 0** والمدقق **valid** فالجدول لا يحوي أي تعارض قاعة أو مدرّس أو طلاب.
القيمة Fitness = 1000 × (مخالفات صارمة) + (عقوبات مرنة)؛ كلما قلّت كان أفضل.
المجموعة **lecture/large** صعبة بنيويًا (63% من أزواج الأحداث تتشارك طلابًا) فقد تبقى فيها 14–21 مخالفة؛ هذه نتيجة موثّقة في التقرير.

## 4) تشغيل بقية الأجزاء

```bash
python -m unittest discover -s tests -t .        # 49 اختبارًا (أقل من دقيقة)
python run_ga.py                                  # GA على lecture/small ويحفظ الجدول في results/schedules
python run_pso.py
python run_aco.py
python -m experiments.run_all_experiments all     # إعادة كل التجارب (lecture + exam، 3 خوارزميات، 3 أحجام، 3 بذور) — تستغرق وقتًا
python -m data.generate_data                      # إعادة توليد البيانات (حتمية)
python -m data.describe_data                      # تقرير جودة البيانات والخطوط المرجعية
```

مخرجات التجارب في `results/`: `runs_<mode>.csv` (كل تشغيل) و`convergence_<mode>.csv` (منحنيات التقارب) و`analysis.json` و`figures/`.

## 5) هيكل المشروع

| المسار | المحتوى |
|---|---|
| `models/` | الكيانات والجدول `Schedule` |
| `core/` | المسألة `Problem` والقيود ودالة اللياقة والمدقق وخطوط الأساس |
| `algorithms/` | `genetic_algorithm.py` و`particle_swarm.py` و`ant_colony.py` والمشترك `common.py` |
| `data/` | مولّد البيانات والبيانات الست |
| `experiments/` | مشغّل التجارب الآلي |
| `tests/` | 49 اختبارًا |
| `results/` | نتائج التجارب والرسوم |
| `app.py` | واجهة Streamlit |
| `run_ga.py` `run_pso.py` `run_aco.py` | تشغيل سريع لكل خوارزمية |

## 6) الخوارزميات باختصار

- **GA:** مجتمع 40 جدولًا؛ انتخاب بالبطولة (3)، تزاوج منتظم (0.8)، طفرة (0.05 لكل حدث)، نخبوية 1؛ 100 جيل؛ إصلاح Lamarckian للأبناء المخالفين.
- **PSO:** سرب 30 جسيمًا؛ كل جسيم متجه فهارس (وقت، قاعة) لكل حدث؛ w=0.7، c1=c2=1.5؛ 100 تكرار؛ فك الترميز بالتقريب ثم إصلاح وإعادة ترميز.
- **ACO:** 12 نملة؛ 30 تكرارًا؛ فيرومون على (وقت، قاعة) لكل حدث؛ α=1، β=2، تبخر 0.3؛ قواعد MAX-MIN؛ إصلاح قصير للنملة المخالفة.
- في الثلاث: صقل نهائي (حتى 20000 خطوة) للحل الأفضل إن بقيت مخالفات، ويُسجَّل نقطةً منفصلة في منحنى التقارب (`final_polish`).

الشرح التفصيلي والنتائج والمقارنات في **التقرير** (ملف Word المرفق).

## 7) البيانات

ست مجموعات في `data/<mode>/<size>/` (`lecture`/`exam` × `small`/`medium`/`large`)، كل منها يحوي
`courses.csv`, `rooms.csv`, `teachers.csv`, `students.csv`, `enrollments.csv`, `meta.json`
و`planted_schedule.csv` (جدول صالح معروف مسبقًا). البيانات **اصطناعية** وليست بيانات جامعية حقيقية.

## 8) ملفات الفريق الأول (المحرك)

| الملف | الوظيفة |
|---|---|
| `core/fitness.py` | دالة الهدف المشتركة `evaluate()` |
| `core/validator.py` | المدقق المستقل `validate_schedule()` |
| `core/baseline.py` | خطّا أساس: عشوائي وجشع |
| `data/generate_data.py` | مولّد البيانات بحل مزروع |

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

