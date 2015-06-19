% Release date: June 2015
% Author: Taewoo Lee, (twlee@speech.korea.ac.kr)
%
% Copyright (C) 2015 Taewoo Lee
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program. If not, see <http://www.gnu.org/licenses/>.
%
% Reference:
% [1] Taewoo Lee, Sukmoon Chang, and Dongsuk Yook, "Parallel SRP-PHAT for
%     GPUs," Computer Speech and Language, vol. 35, pp. 1-13, Jan. 2016.
%
clc; clear all; close all;

Fs= 16000; % sampling frequency (Hz)
c= 340000; % speed of sound (mm/s)

% coordinates of a microphone array (cartesian coordinates), milli meter
load('../../gen_simul_data/mic_array.mat');
mic_array= mic_array.*1000;     % meter -> milli meter
mic_array_origin= [0 0 400];    % in the room
n_mic= size(mic_array,1);

cnt=1;
for i=[1 3 11 13 15 17 29 31]  % 8 ch
    mic_array_8ch(cnt,:)= mic_array(i,:);
    cnt= cnt+1;
end
cnt=1;
for i=[1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31]  % 16 ch
    mic_array_16ch(cnt,:)= mic_array(i,:);
    cnt= cnt+1;
end
mic_array_32ch= mic_array;

M= [8 16 32];
Q= [3888 97200 388800];

disp('Make sphCoords.mat and TDOA_table.mat');
result_dir= './mat';
if exist(result_dir,'dir')
    rmdir(result_dir,'s');
end
mkdir(result_dir);
for nm=M
    fprintf('M= %d\n',nm);
    if nm==8
        input_mic_array= mic_array_8ch;
    elseif nm==16
        input_mic_array= mic_array_16ch;
    elseif nm==32
        input_mic_array= mic_array_32ch;
    end
    
    for nq=Q
        fprintf('\tQ= %d\n',nq);
        if nq==3888
            search_range.theta= 0:5:355;        % degree
            search_range.phi= 0:5:85;           % degree
            search_range.r= [1000 2000 3000];   % milli meter
        elseif nq==97200
            search_range.theta= 0:1:359;        % degree
            search_range.phi= 0:1:89;           % degree
            search_range.r= [1000 2000 3000];   % milli meter
        elseif nq==388800
            search_range.theta= 0:0.5:359.5;    % degree
            search_range.phi= 0:0.5:89.5;       % degree
            search_range.r= [1000 2000 3000];   % milli meter
        end
        
        mk_tdoa_table_full(input_mic_array,mic_array_origin,search_range,Fs,c);
        
        dest_filename= sprintf('sphCoords_Q%d.mat',nq);
        movefile('sphCoords.mat',[result_dir '/' dest_filename]);
        dest_filename= sprintf('TDOA_table_M%d_Q%d.mat',nm,nq);
        movefile('TDOA_table.mat',[result_dir '/' dest_filename]);
    end
end

disp('*.mat -> *.bin');
result_dir= './bin';
if exist(result_dir,'dir')
    rmdir(result_dir,'s');
end
mkdir(result_dir);
mat_dir= './mat';
for nm=M
    fprintf('M= %d\n',nm);
    for nq=Q
        fprintf('\tQ= %d\n',nq);
        
        load_filename= sprintf('%s/sphCoords_Q%d.mat',mat_dir,nq);
        load(load_filename);
        load_filename= sprintf('%s/TDOA_table_M%d_Q%d.mat',mat_dir,nm,nq);
        load(load_filename);
        
        sphCoords= cast(sphCoords,'int32')';
        dest_filename= sprintf('sphCoords_Q%d.bin',nq);
        fid= fopen(dest_filename,'wb');
        fwrite(fid,sphCoords,'int32','ieee-le');    % For Linux
        fclose(fid);
        movefile(dest_filename,[result_dir '/' dest_filename]);
        
        TDOA_table= cast(TDOA_table,'int16')';
        dest_filename= sprintf('TDOA_table_M%d_Q%d.bin',nm,nq);
        fid= fopen(dest_filename,'wb');
        fwrite(fid,TDOA_table,'int16','ieee-le');    % For Linux
        fclose(fid);
        movefile(dest_filename,[result_dir '/' dest_filename]);
    end
end
