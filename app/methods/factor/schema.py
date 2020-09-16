from typing import Optional
from pydantic import BaseModel, Field, HttpUrl
from app.core.schema import ImageFormatEnum, ImageDpiEnum, FileFormatEnum


class PcaTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    ncomponents: int = Field(2, gt=1, title='Количество компонент')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')

class PcaChooseTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')


class FactorAnalysisTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    ncomponents: int = Field(2, gt=1, title='Количество факторов')
    file_format: FileFormatEnum = Field(FileFormatEnum.CSV, title='Формат выходных табличных файлов')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')


class FactorScreePlotTaskParams(BaseModel):
    url: HttpUrl
    columns: str = Field('1', regex=r'(^\d+(-\d+)?(?:,\d+(?:-\d+)?)*$)|(^\*$)', title='Столбцы для расчёта', description='Пример: 2-4,6,8')
    image_format: ImageFormatEnum = Field(ImageFormatEnum.JPG, title='Формат выходных изображений')
    image_dpi: ImageDpiEnum = Field(ImageDpiEnum.DPI_300, title='Качество выходных изображений (DPI)')
