-- scripts/init-db.sql
-- Database initialization script for Dana AI Copilot

-- Ensure proper character set
SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- Grant privileges to dana user
GRANT ALL PRIVILEGES ON dana.* TO 'dana'@'%';
FLUSH PRIVILEGES;

-- Create placeholder tables if they don't exist from Prisma
-- These will be replaced by Prisma migrations in production

-- Note: In production, the existing CanApply tables (students, funding_requests, etc.)
-- would already exist and be managed by the Laravel application.
-- This script only creates minimal development fixtures.

-- Development seed data will be inserted by a separate script or Prisma seed





