terraform {
    required_providers {
      google = {
        source  = "hashicorp/google"
        version = "~> 4.0"
      }
    }
  }

  provider "google" {
    credentials = file("../service-account-key.json")
    project     = "<gcp_project_id>"
    region      = "us-central1"
  }

  # Create Google Cloud Storage bucket
  resource "google_storage_bucket" "ml_bucket" {
    name     = "<gcp_bucket_name>"
    location = "us-central1"
    force_destroy = false
    uniform_bucket_level_access = true
  }

  # Create Artifact Registry repository
  resource "google_artifact_registry_repository" "ml_repository" {
    location      = "us-central1"
    repository_id = "<gcp_artifact_registry_name>"
    format        = "DOCKER"
    description   = "Docker repository for in-context-learning ML models"
  }