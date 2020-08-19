from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field, HttpUrl
from app.core.schema import ImageFormatEnum, ImageDpiEnum, FileFormatEnum

class StrategyEnum(str, Enum):
    MEAN = 'Среднее'
    MEDIAN = 'Медиана'
    MOST_FREQUENT = 'Наиболее частое'
    CONSTANT = 'Константа'

class MissingValuesTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    missing_value: float = Field(-1, title='Величина, обозначающая пропущенное значение')
    strategy: StrategyEnum = Field(StrategyEnum.MEAN, title='Стратегия замены (на что заменять)')
    fill_value: float = Field(0, title='Значение константы (если стратения замены - Констранта)')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')

class NormalizationTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    lower_bound: float = Field(0, title='Нижняя граница диапазона')
    upper_bound: float = Field(1, title='Верхняя граница диапазона')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')

class StandartizationTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
