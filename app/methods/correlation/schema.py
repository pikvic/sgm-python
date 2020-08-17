from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from app.core.schema import ImageFormatEnum, ImageDpiEnum, FileFormatEnum


class CorrmatrixTaskParams(BaseModel):
    url: HttpUrl
    columns1: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Первый набор столбцов', description='Пример: 2-4,6,8')
    columns2: str = Field('2', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Второй набор столбцов', description='Пример: 2-4,6,8')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    #image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    #image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')

class ScatterplotTaskParams(BaseModel):
    url: HttpUrl
    columns1: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Первый набор столбцов', description='Пример: 2-4,6,8')
    columns2: str = Field('2', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Второй набор столбцов', description='Пример: 2-4,6,8')
    #file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')
