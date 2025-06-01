import setuptools

setuptools.setup(
    name="nnunetv2",
    version="2.6.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        # Add dependencies here, e.g.
        # "numpy",
        # "torch",
    ],
    entry_points={
        "console_scripts": [
            "nnUNetv2_plan_and_preprocess=nnunetv2.experiment_planning.plan_and_preprocess_entrypoints:entry_point",
            "nnUNetv2_predict=nnunetv2.inference.predict_from_raw_data:entry_point",
        ],
    },
)
