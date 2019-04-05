function [center_set] = get_center_coords(im)

[r,c] = find(im>0);
center_set = [c,r];

end