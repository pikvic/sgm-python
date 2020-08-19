from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from app.core.schema import ImageFormatEnum, ImageDpiEnum, FileFormatEnum

class KMeansTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    nclusters: int = Field(6, gt=1, le=30, title='Количество кластеров')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    #image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    #image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')

class KMeansScreePlotTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    max_clusters: int = Field(20, gt=1, le=30, title='Максимальное количество кластеров')
    #file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')
