function []=mk_tdoa_table_full(mic_array,mic_array_origin,search_range,Fs,c)
% In : mic_array: cartesian coordinates of microphones
%      mic_array_origin: origin of the microphone array
%      search_range: [theta, phi, r] range to search
%      Fs: sampling frequency (Hz).
%      c: speed of sound (mm/s)
% Out: TDOA_table.mat: Q x N matrix which stores TDOAs of each microphone 
%                      pair in N for each coordinates in Q.
%      sphCoords.mat : spherical coordinates of each coordinates
%                      (Q x 3(r,theta,phi)).
% Reference:
% [2] J. Dmochowski, J. Benesty, and S. Affes, "A generalized steered response 
%     power method for computationally viable source localization," 
%     IEEE Transactions on Audio, Speech, and Language Processing, vol. 15, 
%     pp. 2510-2526, 2007.
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

TDOA_table= zeros(coordCnt,N);
for i=1:coordCnt
    micPairCnt= 0;
    for m1=1:M-1
        for m2=m1+1:M
            d1= norm(mic_array(m1,:)-cartCoords(i,:),2);
            d2= norm(mic_array(m2,:)-cartCoords(i,:),2);
            dd= d1-d2;
            sd= round((dd/c)*Fs);
            micPairCnt= micPairCnt+1;
            TDOA_table(i,micPairCnt)= sd;
        end
    end
end
save('TDOA_table.mat','TDOA_table');
