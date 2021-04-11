from distutils.core import setup

setup(
    name="driver",
    version="0.1",
    packages=[
        "driver",
        "driver.gym",
        "driver.car",
        "driver.legacy",
        "driver.planner",
    ],
    install_requires=["numpy", "gym", "matplotlib", "pyglet>=1.4", "moviepy", "tensorflow>=2.0"],
    package_data={
        "driver": ["py.typed"],
    },
)
