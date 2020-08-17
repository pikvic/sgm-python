from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from app.core.schema import ImageFormatEnum, ImageDpiEnum, FileFormatEnum

class StatsTaskParams(BaseModel):
    url: HttpUrl
    column: int = Field(..., gt=0)
    transpose: Optional[bool] = False
    showgraph: Optional[bool] = False

class SummaryTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    #image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    #image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')

class HistorgamTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    #file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')

class BoxplotTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    #file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')