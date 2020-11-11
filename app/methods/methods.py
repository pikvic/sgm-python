from app.core.schema import MethodInfo, Group

GROUPS = [
    Group(
        title='Описательная статистика',
        description='Группа методов для исследования входных наборов данных на предмет основных статистических показателей, распределений и др.',
        image='https://ocw.mit.edu/courses/sloan-school-of-management/15-075j-statistical-thinking-and-data-analysis-fall-2011/15-075f11.jpg',
        methods=['summary', 'histogram', 'boxplot']
    ),
    Group(
        title='Корреляционный анализ',
        description='Группа методов для поиска закономерностей в данных.',
        image='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSKOb5EmFODVKcwiUSp8cnCKs47gFzTJhFk7Q&usqp=CAU',
        methods=['corrmatrix', 'scatterplot']
    ),
    Group(
        title='Кластерный анализ',
        description='Методы поиска групп в исходных данных.',
        image='https://img.pngio.com/r-script-showcase-microsoft-power-bi-community-clustering-png-300_173.png',
        methods=['kmeans', 'kmeansscreeplot']
    ),
    Group(
        title='Предобработка данных',
        description='Методы предобработки данных: стандатизация, нормализация, работа с пропущенными значениями и т.д.',
        image='https://w0.pngwave.com/png/160/410/feature-chart-machine-learning-microsoft-azure-data-pre-processing-business-chart-publicity-s-png-clip-art-thumbnail.png',
        methods=['standartization', 'missingvalues', 'normalization']
    ),
    Group(
        title='Факторный анализ',
        description='Методы факторного анализа: выбор количества факторов, факторный анализ, метод главных компонент и т.д.',
        image='https://scikit-learn.org/stable/_images/sphx_glr_plot_pca_iris_001.png',
        methods=['factoranalysis', 'factorscreeplot', 'pca', 'pcachoose']
    ),
    Group(
        title='Регрессионный анализ',
        description='Методы регрессионного анализа: линейная регрессия, полиномиальная регрессия и т.д.',
        image='https://hackernoon.com/images/h31rz24si.jpg',
        methods=['linearregression', 'polynomialregression']
    )
]

METHODS = {
    'summary': MethodInfo(
        group='Описательная статистика',
        title='Основные статистические показатели',
        description='Вычисление основных статистических показателей по выбранным столбцам: среднее, минимум, максимум, квантили, дисперсия и т.д.',
        name='summary',
        image='https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Standard_deviation_diagram.svg/1200px-Standard_deviation_diagram.svg.png'
    ),
    'histogram': MethodInfo(
        group='Описательная статистика',
        title='Гистограмма распределения',
        description='Построение гистограмм распределения для визуальной оценки распределения значений по выбранным столбцам.',
        name='histogram',
        image='https://www.mathworks.com/help/examples/stats/win64/HistogramwithaNormalDistributionFitExample_01.png'
    ),
    'boxplot': MethodInfo(
        group='Описательная статистика',
        title='Диаграмма размаха',
        description='Построение диаграмм размаха для визуальной оценки основных статистических показателей по выбранным столбцам.',
        name='boxplot',
        image='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTdme_orKceyOsu6WIw2XL92DOpK4gZo19sVg&usqp=CAU'
    ),
    'corrmatrix': MethodInfo(
        group='Корреляционный анализ',
        title='Корреляционная матрица',
        description='Вычисление корреляционной матрицы для выбранных наборов столбцов.',
        name='corrmatrix',
        image='https://i.stack.imgur.com/9pa7S.jpg'
    ),
    'scatterplot': MethodInfo(
        group='Корреляционный анализ',
        title='Диаграмма рассеяния',
        description='Построение диаграмм рассеяния для визуальной оценки корреляций для выбранных наборов столбцов.',
        name='scatterplot',
        image='static/scatterplot.png'
    ),
    'kmeans': MethodInfo(
        group='Кластерный анализ',
        title='Метод k-средних',
        description='Кластеризация входных данных методом k-средних. Рекомендуется предварительная нормализация или стандартизация данных.',
        name='kmeans',
        image='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRpg-UkIRFomhPvbHkXizOfXznb9S1FQbv7pw&usqp=CAU'
    ),
    'kmeansscreeplot': MethodInfo(
        group='Кластерный анализ',
        title='Выбор количества кластеров (каменистая осыпь)',
        description='Построение графика каменистой осыпи для количества кластеров в исходных данных по методу k-средних. Позволяет определить оптимальное количество кластеров. Рекомендуется предварительная нормализация или стандартизация данных.',
        name='kmeansscreeplot',
        image='https://www.researchgate.net/profile/Chirag_Deb/publication/320986519/figure/fig8/AS:560163938422791@1510564898246/Result-of-the-elbow-method-to-determine-optimum-number-of-clusters.png'
    ),
    'standartization': MethodInfo(
        group='Предобработка данных',
        title='Стандартизация',
        description='Приведение данных к распределению со средним 0 и стандартным отклонением 1.',
        name='standartization',
        image='https://miro.medium.com/max/1200/1*dZlwWGNhFco5bmpfwYyLCQ.png'
    ),
    'missingvalues': MethodInfo(
        group='Предобработка данных',
        title='Обработка пропусков в данных',
        description='Обработка пропусков в данных на основе остальных значений согласно выбранной стратегии.',
        name='missingvalues',
        image='https://miro.medium.com/max/2816/1*MiJ_HpTbZECYjjF1qepNNQ.png'
    ),
    'normalization': MethodInfo(
        group='Предобработка данных',
        title='Нормализация',
        description='Приведение данных к нужному диапазону.',
        name='normalization',
        image='https://www.educative.io/api/edpresso/shot/5146765192855552/image/6250600309194752.png'
    ),
    'factoranalysis': MethodInfo(
        group='Факторный анализ',
        title='Факторный анализ',
        description='Метод факторного анализа, показывает факторную нагрузку, а также переводит измерения в пространство факторов',
        name='factoranalysis',
        image='https://res.cloudinary.com/dchysltjf/image/upload/f_auto,q_auto:best/v1554830233/1.png'
    ),
    'factorscreeplot': MethodInfo(
        group='Факторный анализ',
        title='Выбор количества факторов (каменистая осыпь)',
        description='Построение графика каменистая осыпь для собственных значений факторов',
        name='factorscreeplot',
        image='https://res.cloudinary.com/dchysltjf/image/upload/f_auto,q_auto:best/v1554830233/3.png'
    ),
    'pca': MethodInfo(
        group='Факторный анализ',
        title='Метод главных компонент',
        description='Метод главных компонент с вычислением всех необходимых значений.',
        name='pca',
        image='https://amva4newphysics.files.wordpress.com/2016/06/pca.gif'
    ),
     'pcachoose': MethodInfo(
        group='Факторный анализ',
        title='Выбор количества главных компонент',
        description='Построение графика доли информации для определения количества главных компонент.',
        name='pcachoose',
        image='https://user.oc-static.com/upload/2019/04/16/15554174747084_pca3_1.png'
    ),
    'linearregression': MethodInfo(
        group='Регрессионный анализ',
        title='Линейная регрессия',
        description='Расчёт коэффициентов линейной регрессии и других параметров по входным данным.',
        name='linearregression',
        image='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png'
    ),
    'polynomialregression': MethodInfo(
        group='Регрессионный анализ',
        title='Полиномиальная регрессия',
        description='Расчёт коэффициентов полиномиальной регрессии по входным данным.',
        name='polynomialregression',
        image='https://imgs.developpaper.com/imgs/3314166135-5c013a8ecd84b_articlex.png'
    ),
    
}
    
