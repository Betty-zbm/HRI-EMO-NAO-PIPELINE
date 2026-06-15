% Octave CLI entry point (keep separate from extract_covarep_segment.m for MATLAB R2026+).
% Usage:
%   octave --no-gui tools/covarep/extract_covarep_segment_octave.m in.wav out.mat /path/to/covarep

if ~exist('OCTAVE_VERSION', 'builtin')
    error('This script is for Octave only.');
end

args = argv();
if numel(args) < 3
    error(['Usage: octave --no-gui extract_covarep_segment_octave.m ', ...
        'in.wav out.mat /path/to/covarep']);
end

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
extract_covarep_segment(args{1}, args{2}, args{3});
