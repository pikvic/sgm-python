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
        group='Описательная статистика',
        title='Диаграмма рассеяния',
        description='Построение диаграмм рассеяния для визуальной оценки корреляций для выбранных наборов столбцов.',
        name='scatterplot',
        image='static/scatterplot.png'
    ),
    'kmeans': MethodInfo(
        group='Кластерный анализ',
        title='Метод k-средних',
        description='Кластеризация входных данных методом k-средних.',
        name='scatterplot',
        image='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRpg-UkIRFomhPvbHkXizOfXznb9S1FQbv7pw&usqp=CAU'
    ),
    'kmeansscreeplot': MethodInfo(
        group='Кластерный анализ',
        title='Выбор количества кластеров (каменистая осыпь)',
        description='Построение графика каменистой осыпи для количества кластеров в исходных данных по методу k-средних. Позволяет определить оптимальное количество кластеров.',
        name='kmeansscreeplot',
        image='https://www.researchgate.net/profile/Chirag_Deb/publication/320986519/figure/fig8/AS:560163938422791@1510564898246/Result-of-the-elbow-method-to-determine-optimum-number-of-clusters.png'
    ),
}
    
