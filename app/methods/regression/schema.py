from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from app.core.schema import ImageFormatEnum, ImageDpiEnum, FileFormatEnum


class LinearRegressionTaskParams(BaseModel):
    url: HttpUrl
    target_column: int = Field(2, gt=1, title='Зависимая переменная (целевая)')
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы с независимыми переменными', description='Пример: 2-4,6,8')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')

class PolynomialRegressionTaskParams(BaseModel):
    url: HttpUrl
    target_column: int = Field(2, gt=1, title='Зависимая переменная (целевая)')
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы с независимыми переменными', description='Пример: 2-4,6,8')
    degree: int = Field(1, gt=0, le=10, title='Степень полинома')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    
