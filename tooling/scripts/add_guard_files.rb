#!/usr/bin/env ruby
# Adds NotchNerd/Guard/*.swift to a Guard PBXGroup + the app target's Sources phase, and links the
# PresenceKit package (local path, ../presence-lock relative to the repo by default). Idempotent.
#
# Usage (from repo root):
#   ruby tooling/scripts/add_guard_files.rb [path/to/presence-lock]
require "xcodeproj"

PROJECT     = File.expand_path("NotchNerd.xcodeproj", Dir.pwd)
TARGET_NAME = "NotchNerd"
PKG_PATH    = ARGV[0] || "../presence-lock"

project = Xcodeproj::Project.open(PROJECT)
target  = project.targets.find { |t| t.name == TARGET_NAME } or abort "target #{TARGET_NAME} not found"
app     = project.main_group.find_subpath("NotchNerd", false) or abort "NotchNerd group not found"
group   = app.children.find { |c| c.display_name == "Guard" } || app.new_group("Guard", "Guard")

Dir.glob(File.join(File.dirname(PROJECT), "NotchNerd/Guard/*.swift")).sort.each do |disk|
  name = File.basename(disk)
  ref = group.files.find { |f| f.display_name == name } || group.new_reference(name)
  target.source_build_phase.add_file_reference(ref, true) unless target.source_build_phase.files_references.include?(ref)
  puts "guard file: #{name}"
end

pkg = project.root_object.package_references.find { |p| p.respond_to?(:relative_path) && p.relative_path == PKG_PATH }
unless pkg
  pkg = project.new(Xcodeproj::Project::Object::XCLocalSwiftPackageReference)
  pkg.relative_path = PKG_PATH
  project.root_object.package_references << pkg
  puts "added local package #{PKG_PATH}"
end
unless target.package_product_dependencies.any? { |d| d.product_name == "PresenceKit" }
  dep = project.new(Xcodeproj::Project::Object::XCSwiftPackageProductDependency)
  dep.product_name = "PresenceKit"
  dep.package = pkg
  target.package_product_dependencies << dep
  bf = project.new(Xcodeproj::Project::Object::PBXBuildFile)
  bf.product_ref = dep
  target.frameworks_build_phase.files << bf
  puts "linked PresenceKit"
end
project.save
