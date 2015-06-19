function []=mk_toa_table_full(mic_array,mic_array_origin,search_range,Fs,c)
% In : mic_array: cartesian coordinates of microphones
%      mic_array_origin: origin of the microphone array
%      search_range: [theta, phi, r] range to search
%      Fs: sampling frequency (Hz).
%      c: speed of sound (mm/s)
% Out: TOA_table.mat:  Q x M matrix which stores TOAs of each microphone 
%                      in M for each coordinates in Q.
%      sphCoords.mat : spherical coordinates of each coordinates
%                      (Q x 3(r,theta,phi)).
%
% Reference:
% [1] Taewoo Lee, Sukmoon Chang, and Dongsuk Yook, "Parallel SRP-PHAT for
%     GPUs," Computer Speech and Language, vol. 35, pp. 1-13, Jan. 2016.
%

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
M= size(mic_array,1); % number of microphones
N= M*(M-1)/2; % number of microphone pairs

n_theta= size(search_range.theta,2);
n_phi= size(search_range.phi,2);
n_r= size(search_range.r,2);
cartCoords= zeros(n_theta*n_phi*n_r, 3);
sphCoords= zeros(n_theta*n_phi*n_r, 3);
coordCnt= 0;
for r=search_range.r
    for ti=search_range.theta
        for pi=search_range.phi;
            [x(1),x(2),x(3)]= sph2cart(deg2rad(ti),deg2rad(pi),r);
            coordCnt= coordCnt+1;
            cartCoords(coordCnt,:)= x + mic_array_origin;
            sphCoords(coordCnt,:)= [r,ti,pi];
        end
    end
end
% save('cartCoords.mat','cartCoords','coordCnt');
save('sphCoords.mat','sphCoords','coordCnt');

TOA_table= zeros(coordCnt,M);
for i=1:coordCnt
    for m=1:M
        d= norm(mic_array(m,:)-cartCoords(i,:),2);
        sd= (d/c)*Fs;
        TOA_table(i,m)= sd;
    end
end
save('TOA_table.mat','TOA_table');
