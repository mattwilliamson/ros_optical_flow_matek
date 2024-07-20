import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'ros_optical_flow_matek'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=[]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyserial', 'numpy'],
    zip_safe=True,
    maintainer='Matt Williamson',
    maintainer_email='matt@aimatt.com',
    description='ROS 2 package for the Matek 3901-l0x Optical Flow Sensor',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'optical_flow_publisher = ros_optical_flow_matek.optical_flow_publisher:main',
            'optical_flow_node = ros_optical_flow_matek.optical_flow_node:main',
        ],
    },
)
