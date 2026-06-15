function w = hanning(n)
%HANNING R2026+ shim: COVAREP calls hanning(); modern MATLAB provides hann().
    w = hann(n);
end
