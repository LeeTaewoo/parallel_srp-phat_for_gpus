function []=frameExtracter(filelist,M,T)
% frameExtractor_simul_mat.m, Taewoo Lee, 2014.7.31.
% IN: ld_vadlist.m, enframe.m
% OUT: VADed_dat.mat

% Release date: May 2015
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
filelist= ld_vadlist();
nfile= size(filelist,1);

disp('Extract VAD frames');
% Extract VADed frames, then save it as *.mat files
for i=1:nfile
    fprintf('Processing %s...\n',filelist{i,1});
    fnlen= size(filelist{i,1},2);
    vadfn= [filelist{i,1}(1:fnlen-4) '_VAD.mat'];
    load(vadfn); % load VAD result on results
    
    % enframed M channel wavfiles
    f= cell(M,1);
    for j=1:M
        read_filename= [filelist{i,1}(1:fnlen-5) sprintf('%d.wav',j-1)];
        [x(:,j),Fs]= wavread(read_filename);
        f{j,1}= enframe(x(:,j),hamming(T,'periodic'),T/2);
    end
    
    extract_frames= find(results==1);
    f2= cell(M,1);
    for m=1:M
        f2{m,1}= f{m,1}(extract_frames,:);
        f2{m,1}= cast(f2{m,1}'*32767,'int16');
    end
    
    % save it in *.mat format. (For user readability).
    output_filename= [filelist{i,1}(1:fnlen-8) 'VADed_data.mat'];
    save(output_filename,'f2');
    
    % save in binary format. (In little endian, for linux).
    output_filename= sprintf('%sinfo.dat',filelist{i,1}(1:fnlen-8));
    fid= fopen(output_filename,'wb');
    nFrames= size(f2{1,1},2);
    dimension=cast([nFrames; T],'int32');
    fwrite(fid,dimension,'int32','ieee-le');
    fclose(fid);
    for m=1:M
        output_filename= sprintf('%s%d.dat',filelist{i,1}(1:fnlen-8),m-1);
        fid= fopen(output_filename,'wb');
        fwrite (fid,f2{m,1},'int16','ieee-le');
        fclose(fid);
    end
end