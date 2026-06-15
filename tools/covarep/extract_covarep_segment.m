function extract_covarep_segment(wav_in, mat_out, covarep_root)
%EXTRACT_COVAREP_SEGMENT Extract MOSEI-compatible 74-d COVAREP frame features.
%
% Uses COVAREP's COVAREP_feature_formant_extraction_perfile with the common
% 74-dimensional feature set used in CMU-MOSEI (COVAREP view):
%   f0, VUV, NAQ, QOQ, H1H2, PSP, MDQ, HRF, peakSlope, Rd, Rd_conf,
%   MCEP_0..24, HMPDM_0..24, HMPDD_0..12  => 74 dims @ 100 Hz
%
% Args:
%   wav_in       - Input mono WAV path (16 kHz recommended)
%   mat_out      - Output .mat path with variable ``features`` [T, 74]
%   covarep_root - Path to cloned https://github.com/covarep/covarep

    if nargin < 3
        error('Usage: extract_covarep_segment(wav_in, mat_out, covarep_root)');
    end

    if ~exist(wav_in, 'file')
        error('Input WAV not found: %s', wav_in);
    end
    if ~exist(covarep_root, 'dir')
        error('COVAREP root not found: %s', covarep_root);
    end

    startup_path = fullfile(covarep_root, 'startup.m');
    if ~exist(startup_path, 'file')
        error('COVAREP startup.m not found under: %s', covarep_root);
    end

    % Use COVAREP's startup only — do NOT genpath() first. genpath leaves
    % external/backcompatibility_2015/audioread.m (wavread shim) on the path
    % even after startup, which breaks MATLAB R2015+ where wavread was removed.
    script_dir = fileparts(mfilename('fullpath'));
    addpath(script_dir);
    run(startup_path);
    addpath(fullfile(script_dir, 'compat'), '-begin');

    if ~exist('lpc', 'file')
        error(['Signal Processing Toolbox is required for COVAREP (missing lpc). ', ...
            'Install it via MATLAB Add-Ons.']);
    end
    if ~exist('skewness', 'file')
        error(['Statistics and Machine Learning Toolbox is required for COVAREP ', ...
            '(missing skewness). Install it via MATLAB Add-Ons.']);
    end

    options = struct();
    options.feature_fs = 0.01;  % 100 Hz frame rate (MOSEI COVAREP)
    options.features = {'f0', 'VUV', 'NAQ', 'QOQ', 'H1H2', 'PSP', 'MDQ', ...
        'HRF', 'peakSlope', 'Rd', 'Rd_conf', 'MCEP', 'HMPDM', 'HMPDD'};
    options.mcep_order = 24;
    options.hmpdm_order = 24;
    options.hmpdd_order = 12;
    options.save_mat = false;
    options.save_csv = false;

    extractor = fullfile(covarep_root, 'feature_extraction', ...
        'COVAREP_feature_formant_extraction_perfile.m');
    if ~exist(extractor, 'file')
        error('COVAREP extractor not found: %s', extractor);
    end

    results = COVAREP_feature_formant_extraction_perfile(wav_in, options);

    feat_names = results.Properties.VariableNames;
    feat_names = feat_names(~strcmp(feat_names, 'time'));
    features = table2array(results(:, feat_names));

    if isempty(features)
        features = zeros(1, 74);
    end

    features(~isfinite(features)) = 0;

    target_dim = 74;
    [n_frames, n_dims] = size(features);
    if n_dims > target_dim
        features = features(:, 1:target_dim);
    elseif n_dims < target_dim
        pad = zeros(n_frames, target_dim - n_dims);
        features = [features, pad];
    end

    feat_names = feat_names(1:min(numel(feat_names), target_dim));
    if numel(feat_names) < target_dim
        for k = (numel(feat_names) + 1):target_dim
            feat_names{k} = sprintf('pad_%d', k);
        end
    end

    out_dir = fileparts(mat_out);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    save(mat_out, 'features', 'feat_names', '-v7');
    fprintf('[COVAREP] Saved %dx%d features to %s\n', size(features, 1), size(features, 2), mat_out);
end
