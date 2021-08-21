from setuptools import setup

setup(name='ml_me',
      version='0.2',
      description='The funniest joke in the world',
      url='https://github.com/stauntonmakerspace/StauntonMakerSign',
      author='Nile Walker',
      author_email='nilezwalker@gmail.com',
      license='MIT',
      packages=['ml_me'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)