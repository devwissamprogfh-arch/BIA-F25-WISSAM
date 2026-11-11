# اتفاق الفرق قبل البدء (Team Agreement)

الخطة تنص على أن الفرق تتفق أولًا على: صيغة البيانات، واجهة الدوال، وطريقة حساب القيود.
هذا الملف يوثّق ما سلّمه الفريق الأول، ويقترح عقدًا لاستدعاء الخوارزميات.

## 1) صيغة البيانات
CSV + `meta.json` كما في `README.md`. الزمن مقسّم إلى فترات (slots) برقم عام
`g = day * slots_per_day + slot_in_day`. الحدث الذي يبدأ عند `g` ومدته `d` يشغل `[g, g+d)`،
وحدثان متتاليان (نهاية الأول = بداية الثاني) **لا** يتعارضان.

## 2) تمثيل الحل
`Schedule(slots, rooms)`: قائمتان صحيحتان محاذيتان لـ `problem.events`.
القيمة `-1` تعني «غير مُسند». يمكن تحويلها إلى متجه `to_vector()` وإعادتها `from_vector()`.

| الخوارزمية | الاستعمال المقترح |
|---|---|
| GA | الجين i = `(slots[i], rooms[i])`؛ الطفرة تختار قيمة من `valid_starts[i]` و`feasible_rooms[i]` |
| PSO | موضع الجسيم = المتجه المسطّح؛ التقريب والقص إلى المجال الصحيح قبل التقييم |
| ACO | النملة تُسند حدثًا واحدًا في كل خطوة؛ المرشحات من `valid_starts` و`feasible_rooms`؛ المعلومة الإرشادية من `neighbors[i]` |

## 3) واجهة التقييم (إلزامية للجميع)
```python
fitness(problem, schedule, config=None) -> float          # lower is better
evaluate(problem, schedule, config=None) -> FitnessResult # counts per constraint + totals
validate_schedule(problem, schedule) -> ValidationReport  # independent final check
```
ممنوع على أي فريق تعريف دالة هدف خاصة به.

## 4) عقد الخوارزمية المقترح (للاتفاق عليه مع فريقي الخوارزميات والتكامل)
```python
def run(problem, params: dict, seed: int, max_iterations: int) -> dict:
    """Return a dict with:
    'best_schedule'  : Schedule      the best schedule found
    'best_fitness'   : float         fitness(problem, best_schedule)
    'history'        : list[float]   best fitness after every iteration (for convergence plots)
    'evaluations'    : int           number of fitness calls (fair-budget comparison)
    'time_seconds'   : float         wall-clock time
    """
```

## 5) طريقة حساب القيود
وحدات العدّ مذكورة في جدول القيود (`README.md`) وفي `core/constraints.py`،
وتطابقها حرفيًا في `fitness.py` و`validator.py` (يتحقق من ذلك اختبار مقارنة آلي).

## 6) مراجع زمنية مفيدة (لتقدير ميزانية التجارب)
زمن استدعاء واحد لدالة التقييم: ~0.1 مللي ثانية (small)، ~0.4 (medium)، ~1.2 (large) على جهازنا.
