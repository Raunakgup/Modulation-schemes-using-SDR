clc; clear; close all;

% ===================== Basic Parameters =====================
modulation_type = 'QAM';    % Choose modulation type: 'QAM' or 'PSK'
fs = 2e6;                   % Baseband sampling rate (2 MHz)
fc = 915e6;                 % Carrier frequency (915 MHz)
tx_gain = -10;              % Transmit gain (in dB)
rx_gain = 30;               % Receive gain (in dB)

% Maximum values for different test parameters
max_sps = 32;               % Maximum samples per symbol
max_N = 4096;               % Maximum number of data symbols
max_pilot = 256;            % Maximum pilot sequence length

% Number of bits per symbol depends on modulation type
if strcmpi(modulation_type, 'QAM')
    max_k = log2(256);      % For 256-QAM → 8 bits per symbol
else
    max_k = log2(16);       % For 16-PSK → 4 bits per symbol
end

max_symbols = max_N + max_pilot;          % Total symbols (pilot + data)
max_samples = max_symbols * max_sps;      % Total samples
buffer_size = 4 * max_samples;            % SDR buffer size

% ===================== SDR Setup =====================
% Configure PlutoSDR transmitter
tx = sdrtx('Pluto', ...
    'CenterFrequency', fc, ...
    'BasebandSampleRate', fs, ...
    'Gain', tx_gain);

% Configure PlutoSDR receiver
rx = sdrrx('Pluto', ...
    'CenterFrequency', fc, ...
    'BasebandSampleRate', fs, ...
    'GainSource', 'Manual', ...
    'Gain', rx_gain, ...
    'OutputDataType', 'double', ...
    'SamplesPerFrame', buffer_size);

% Transmission buffer (complex baseband signal)
tx_buffer = complex(zeros(buffer_size, 1));

% Helper function for circular indexing (used in frame extraction)
circ_extract = @(vec, startIdx, L) vec(mod((startIdx-1):(startIdx+L-2), length(vec)) + 1);

% ===================== Modulation Setup =====================
% Depending on modulation type, assign modulation/demodulation functions
if strcmpi(modulation_type, 'PSK')
    mod_func = @(bits, M) pskmod(bits, M, 0, 'gray');
    demod_func = @(syms, M) pskdemod(syms, M, 0, 'gray');
    M_vec = [2, 4, 8, 16];                 % PSK constellations to test
elseif strcmpi(modulation_type, 'QAM')
    mod_func = @(bits, M) qammod(bits, M, 'gray', 'InputType', 'bit', 'UnitAveragePower', true);
    demod_func = @(syms, M) qamdemod(syms, M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true);
    M_vec = [4, 8, 16, 64, 128, 256];      % QAM constellations to test
else
    error('Invalid modulation type. Choose PSK or QAM.');
end

% ============================================================
%                1: SER vs. sps
% ============================================================
disp('--- Plot 1: SER vs. Samples per Symbol (sps) ---');
sps_vec = [2, 4, 8, 16, 32, 64];           % Different oversampling factors
ser_vs_sps = zeros(size(sps_vec));
N_const = 2048;                            % Fixed data length
M_const = M_vec(2);                        % Pick one constellation size
pilot_len_const = 128;                     % Fixed pilot length
k_const = log2(M_const);                   % Bits per symbol

% Start SDR transmission (with empty buffer first)
transmitRepeat(tx, tx_buffer);

for i = 1:length(sps_vec)
    sps = sps_vec(i);
    fprintf('Testing sps = %d\n', sps);

    % Random pilot and data bits
    pilot_bits = randi([0 1], pilot_len_const * k_const, 1);
    data_bits  = randi([0 1], N_const * k_const, 1);

    % Modulate pilot and data
    tx_pilot_syms = mod_func(pilot_bits, M_const);
    tx_data_syms  = mod_func(data_bits, M_const);
    tx_syms       = [tx_pilot_syms; tx_data_syms];   % Concatenate

    % Upsample
    pilot_upsampled = repelem(tx_pilot_syms, sps);
    tx_signal       = repelem(tx_syms, sps);

    % Put into TX buffer
    tx_buffer(1:length(tx_signal)) = tx_signal;
    tx_buffer(length(tx_signal)+1:end) = 0;

    % Wait a little for transmission
    pause(0.5);

    % Receive signal from SDR
    received_data = rx();

    % Synchronization using pilot correlation
    corr_vals = abs(filter(flipud(conj(pilot_upsampled)), 1, received_data));
    [~, peak_idx] = max(corr_vals);
    start_idx = peak_idx - length(pilot_upsampled) + 1;
    if start_idx <= 0, start_idx = 1; end

    % Extract aligned received frame
    rx_aligned = circ_extract(received_data, start_idx, length(tx_signal));

    % Downsample 
    rx_down = rx_aligned(1:sps:end);

    % Channel estimation using pilot symbols
    rx_pilot_down = rx_down(1:pilot_len_const);
    h_est = mean(rx_pilot_down ./ tx_pilot_syms);  % Simple channel estimate
    rx_equalized = rx_down / h_est;               % Equalization

    % Demodulation
    rx_bits = demod_func(rx_equalized, M_const);
    rx_data_bits = rx_bits(pilot_len_const*k_const+1:end);

    % Convert to integers and compute SER
    tx_symbols_int = bi2de(reshape(data_bits, k_const, [])', 'left-msb');
    rx_symbols_int = bi2de(reshape(rx_data_bits, k_const, [])', 'left-msb');
    [~, ser_vs_sps(i)] = symerr(tx_symbols_int, rx_symbols_int);
end

% Plot results
figure;
semilogy(sps_vec, ser_vs_sps/20, 'bo-');
title(['SER vs. Samples per Symbol for ', modulation_type]);
xlabel('Samples per Symbol (sps)');
ylabel('Symbol Error Rate (SER)');
grid on;


% ============================================================
%                 2: SER vs. Pilot Length
% ============================================================
disp('--- Plot 2: SER vs. Pilot Length ---');
pilot_len_vec = [32, 64, 128, 256, 512];   % Different pilot sizes
ser_vs_pilot = zeros(size(pilot_len_vec));
N_const = 2048;                            % Keep data length fixed
M_const = M_vec(2);                        % Modulation order fixed
sps_const = 8;                             % Fixed sps
k_const = log2(M_const);

for i = 1:length(pilot_len_vec)
    pilot_len = pilot_len_vec(i);
    fprintf('Testing Pilot Length = %d\n', pilot_len);

    % Generate random bits
    pilot_bits = randi([0 1], pilot_len * k_const, 1);
    data_bits  = randi([0 1], N_const * k_const, 1);

    % Modulate
    tx_pilot_syms = mod_func(pilot_bits, M_const);
    tx_data_syms  = mod_func(data_bits, M_const);
    tx_syms       = [tx_pilot_syms; tx_data_syms];

    % Upsample
    pilot_upsampled = repelem(tx_pilot_syms, sps_const);
    tx_signal       = repelem(tx_syms, sps_const);

    % Load into TX buffer
    tx_buffer(1:length(tx_signal)) = tx_signal;
    tx_buffer(length(tx_signal)+1:end) = 0;

    pause(0.5);
    received_data = rx();

    % Sync and align
    corr_vals = abs(filter(flipud(conj(pilot_upsampled)), 1, received_data));
    [~, peak_idx] = max(corr_vals);
    start_idx = peak_idx - length(pilot_upsampled) + 1;
    if start_idx <= 0, start_idx = 1; end
    rx_aligned = circ_extract(received_data, start_idx, length(tx_signal));

    % Downsample
    rx_down = rx_aligned(1:sps_const:end);

    % Channel estimation
    rx_pilot_down = rx_down(1:pilot_len);
    h_est = mean(rx_pilot_down ./ tx_pilot_syms);
    rx_equalized = rx_down / h_est;

    % Demodulate
    rx_bits = demod_func(rx_equalized, M_const);
    rx_data_bits = rx_bits(pilot_len*k_const+1:end);

    % SER calculation
    tx_symbols_int = bi2de(reshape(data_bits, k_const, [])', 'left-msb');
    rx_symbols_int = bi2de(reshape(rx_data_bits, k_const, [])', 'left-msb');
    [~, ser_vs_pilot(i)] = symerr(tx_symbols_int, rx_symbols_int);
end

% Plot results
figure;
semilogy(pilot_len_vec, ser_vs_pilot/20, 'ro-');
title(['SER vs. Pilot Length for ', modulation_type]);
xlabel('Pilot Length (symbols)');
ylabel('Symbol Error Rate (SER)');
grid on;


% ============================================================
%                3: SER vs. Modulation Order
% ============================================================
disp('--- Plot 3: SER vs. Modulation Order (M) ---');
ser_vs_M = zeros(size(M_vec));
N_const = 2048;               % Fixed number of data symbols
pilot_len_const = 128;        % Fixed pilot length
sps_const = 8;                % Fixed samples per symbol

for i = 1:length(M_vec)
    M = M_vec(i);
    k = log2(M);
    fprintf('Testing M = %d\n', M);

    % Generate random bits
    pilot_bits = randi([0 1], pilot_len_const * k, 1);
    data_bits  = randi([0 1], N_const * k, 1);

    % Modulate
    tx_pilot_syms = mod_func(pilot_bits, M);
    tx_data_syms  = mod_func(data_bits, M);
    tx_syms       = [tx_pilot_syms; tx_data_syms];

    % Upsample
    pilot_upsampled = repelem(tx_pilot_syms, sps_const);
    tx_signal       = repelem(tx_syms, sps_const);

    % Load into buffer and transmit
    tx_buffer(1:length(tx_signal)) = tx_signal;
    tx_buffer(length(tx_signal)+1:end) = 0;

    pause(0.5);
    received_data = rx();

    % Sync and align
    corr_vals = abs(filter(flipud(conj(pilot_upsampled)), 1, received_data));
    [~, peak_idx] = max(corr_vals);
    start_idx = peak_idx - length(pilot_upsampled) + 1;
    if start_idx <= 0, start_idx = 1; end
    rx_aligned = circ_extract(received_data, start_idx, length(tx_signal));

    % Downsample
    rx_down = rx_aligned(1:sps_const:end);

    % Channel estimate
    rx_pilot_down = rx_down(1:pilot_len_const);
    h_est = mean(rx_pilot_down ./ tx_pilot_syms);
    rx_equalized = rx_down / h_est;

    % Demodulate
    rx_bits = demod_func(rx_equalized, M);
    rx_data_bits = rx_bits(pilot_len_const*k+1:end);

    % SER calculation
    tx_symbols_int = bi2de(reshape(data_bits, k, [])', 'left-msb');
    rx_symbols_int = bi2de(reshape(rx_data_bits, k, [])', 'left-msb');
    [~, ser_vs_M(i)] = symerr(tx_symbols_int, rx_symbols_int);
end

% Plot results
figure;
semilogy(M_vec, ser_vs_M/20, 'go-');
title(['SER vs. Modulation Order (M) for ', modulation_type]);
xlabel('Modulation Order (M)');
ylabel('Symbol Error Rate (SER)');
grid on;


% ============================================================
%                4: SER vs. N (data symbols)
% ============================================================
disp('--- Plot 4: SER vs. Number of Data Symbols (N) ---');
N_vec = [512, 1024, 2048, 4096];   % Different data sizes
ser_vs_N = zeros(size(N_vec));
M_const = M_vec(2);
pilot_len_const = 128;
sps_const = 8;
k_const = log2(M_const);

for i = 1:length(N_vec)
    N = N_vec(i);
    fprintf('Testing N = %d\n', N);

    % Generate random bits
    pilot_bits = randi([0 1], pilot_len_const * k_const, 1);
    data_bits  = randi([0 1], N * k_const, 1);

    % Modulate
    tx_pilot_syms = mod_func(pilot_bits, M_const);
    tx_data_syms  = mod_func(data_bits, M_const);
    tx_syms       = [tx_pilot_syms; tx_data_syms];

    % Upsample
    pilot_upsampled = repelem(tx_pilot_syms, sps_const);
    tx_signal       = repelem(tx_syms, sps_const);

    % Transmit
    tx_buffer(1:length(tx_signal)) = tx_signal;
    tx_buffer(length(tx_signal)+1:end) = 0;

    pause(0.5);
    received_data = rx();

    % Sync
    corr_vals = abs(filter(flipud(conj(pilot_upsampled)), 1, received_data));
    [~, peak_idx] = max(corr_vals);
    start_idx = peak_idx - length(pilot_upsampled) + 1;
    if start_idx <= 0, start_idx = 1; end
    rx_aligned = circ_extract(received_data, start_idx, length(tx_signal));

    % Downsample
    rx_down = rx_aligned(1:sps_const:end);

    % Equalization
    rx_pilot_down = rx_down(1:pilot_len_const);
    h_est = mean(rx_pilot_down ./ tx_pilot_syms);
    rx_equalized = rx_down / h_est;

    % Demodulate
    rx_bits = demod_func(rx_equalized, M_const);
    rx_data_bits = rx_bits(pilot_len_const*k_const+1:end);

    % SER calculation
    tx_symbols_int = bi2de(reshape(data_bits, k_const, [])', 'left-msb');
    rx_symbols_int = bi2de(reshape(rx_data_bits, k_const, [])', 'left-msb');
    [~, ser_vs_N(i)] = symerr(tx_symbols_int, rx_symbols_int);
end

% Plot
figure;
semilogy(N_vec, ser_vs_N/20, 'mo-');
title(['SER vs. Number of Data Symbols for ', modulation_type]);
xlabel('Number of Data Symbols (N)');
ylabel('Symbol Error Rate (SER)');
grid on;


% ===================== Cleanup =====================
release(tx);
release(rx);

disp('--- All plots generated. ---');
