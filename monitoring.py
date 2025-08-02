import os
import argparse

import pandas as pd
from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
from evidently.ui.workspace import Project, CloudWorkspace

import utils

schema = DataDefinition(
    numerical_columns=[
        'amt',
        'lat',
        'long',
        'merch_lat',
        'merch_long',
        'trans_time',
        'month',
    ],
    categorical_columns=[
        'category_food_dining',
        'category_gas_transport',
        'category_grocery_net',
        'category_grocery_pos',
        'category_health_fitness',
        'category_home',
        'category_kids_pets',
        'category_misc_net',
        'category_misc_pos',
        'category_personal_care',
        'category_shopping_net',
        'category_shopping_pos',
        'category_travel',
        'is_fraud',
        'is_fraud_prediction',
    ],
    classification=[
        BinaryClassification(target='is_fraud', prediction_labels='is_fraud_prediction')
    ],
)

EVIDENTLY_API_KEY = os.getenv("EVIDENTLY_API_KEY")
ORG_ID = os.getenv("EVIDENTLY_ORG_ID")


def parse_args():
    """Parse command line arguments for the Evidently project creation script."""
    parser = argparse.ArgumentParser(description="Evidently project creation script")
    parser.add_argument(
        "--reference_data_path",
        type=str,
        default="data/reference_data.csv",
        help="Path to the reference data CSV file",
    )
    parser.add_argument(
        "--current_data_path",
        type=str,
        default="data/predictions.csv",
        help="Path to the current data CSV file",
    )
    return parser.parse_args()


def create_project(ws: CloudWorkspace, project_name: str, org_id: str) -> Project:
    """
    Create a new Evidently project.

    Args:
        ws: The workspace object.
        project_name: The name of the project to create.
        org_id: The organization ID.

    Returns:
        The created project object.
    """

    project = ws.create_project(project_name, org_id=org_id)  # type: ignore
    project.save()
    print(f"Project created with ID: {project.id}")
    with open("project_id.txt", "w", encoding="utf-8") as f:
        f.write(str(project.id))

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset column drift",
            subtitle="Share of drifted columns",
            size="half",
            values=[
                PanelMetric(
                    legend="Share",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "share"},
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Data Drift",
    )

    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset column drift",
            subtitle="Number of drifted columns",
            size="half",
            values=[
                PanelMetric(
                    legend="Count",
                    metric="DriftedColumnsCount",
                    metric_labels={"value_type": "count"},
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Data Drift",
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Recall",
            subtitle="Model recall over time",
            size="half",
            values=[
                PanelMetric(
                    metric="Recall",
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Precision",
            subtitle="Model precision over time",
            size="half",
            values=[
                PanelMetric(
                    metric="Precision",
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Accuracy",
            subtitle="Model accuracy over time",
            size="half",
            values=[
                PanelMetric(
                    metric="Accuracy",
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="F1 Score",
            subtitle="Model F1 score over time",
            size="half",
            values=[
                PanelMetric(
                    metric="F1Score",
                ),
            ],
            plot_params={"plot_type": "line"},
        ),
        tab="Model Performance",
    )

    return project


def main(args):
    """Main function to run the Evidently project creation and data drift analysis."""
    project_name = "credit-card-fraud-detection"
    if not EVIDENTLY_API_KEY or not ORG_ID:
        raise ValueError(
            "EVIDENTLY_API_KEY or EVIDENTLY_ORG_ID environment variables are not set."
        )
    ws = CloudWorkspace(token=EVIDENTLY_API_KEY, url="https://app.evidently.cloud")
    if os.path.exists("project_id.txt"):
        with open("project_id.txt", "r", encoding="utf-8") as f:
            project_id = f.read().strip()
        project = ws.get_project(project_id)
    else:
        project = create_project(ws, project_name, ORG_ID)

    reference_dataset = pd.read_csv(args.reference_data_path)
    reference_dataset = utils.preprocess_data(reference_dataset)
    current_dataset = pd.read_csv(args.current_data_path)
    current_dataset = utils.preprocess_data(current_dataset)
    reference_dataset = Dataset.from_pandas(reference_dataset, data_definition=schema)
    current_dataset = Dataset.from_pandas(current_dataset, data_definition=schema)
    report = Report([DataDriftPreset(), ClassificationPreset()])
    new_report = report.run(
        reference_data=reference_dataset, current_data=current_dataset
    )
    ws.add_run(project.id, new_report, include_data=False)  # type: ignore


if __name__ == "__main__":
    args = parse_args()
    main(args)
