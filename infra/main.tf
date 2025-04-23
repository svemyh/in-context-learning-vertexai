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
    project     = "norse-figure-456022-c6"
    region      = "us-central1"
  }

  # Create Google Cloud Storage bucket
  resource "google_storage_bucket" "ml_bucket" {
    name     = "eecs282-project"
    location = "us-central1"
    force_destroy = false
    uniform_bucket_level_access = true
  }

  # Create Artifact Registry repository
  resource "google_artifact_registry_repository" "ml_repository" {
    location      = "us-central1"
    repository_id = "eecs282"
    format        = "DOCKER"
    description   = "Docker repository for in-context-learning ML models"
  }