# NovaCloud Deployment Policy

## Release Windows
Production deployments at NovaCloud are permitted Monday through Thursday between 09:00 and 16:00 EST. Friday deployments are frozen unless approved by the VP of Engineering for a Severity-1 hotfix.

## Canary & Rollback SLA
Every production release must first deploy to a 5% canary ring for at least 15 minutes. If the error rate exceeds 1.0% or P95 latency exceeds 450 ms, the automated pipeline triggers a rollback within 5 minutes.

## Required Approvals
All pull requests targeting `main` require at least two peer approvals and passing unit and integration tests with at least 80% line coverage.
