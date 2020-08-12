from app.core.schema import MethodInfo

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
        image='https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTdme_orKceyOsu6WIw2XL92DOpK4gZo19sVg&usqp=CAU'
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
    
