from setuptools import setup

setup(
    name='dsutils',
    version='0.01a',
    description='Utilidades para data science',
    author='caiobelfort',
    author_email='caiobelfort90@gmail.com',
    license='GPL',
    packages=['dsutils'],
    zip_safe=False,
    requires=['matplotlib', 'seaborn', 'pandas']
)
