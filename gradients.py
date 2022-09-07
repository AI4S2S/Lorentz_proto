import numpy as np


def wiggle(da, dim, n):
    ''' Shift a DataArray by ``n`` steps along dimension ``dim``.

    Shifted values are wrapped around the globe. This makes sense only for
    longitude. However, for latitude, it will produce non-phyiscal wrapping for
    the edge cases -90/+90 degrees latitude. Therefore, this domain will be
    removed from the output.
    '''
    VALID_LONS_NAMES = ['lon', 'longitude']
    VALID_LAT_NAMES = ['lat', 'latitude']

    if dim.lower() in VALID_LONS_NAMES:
        # Wrap along longitude
        max_degrees = 360
        new_coords = np.mod(n + da.coords[dim], max_degrees)
        return da.assign_coords({dim: new_coords}).sortby(dim)
    elif dim.lower() in VALID_LAT_NAMES:
        # Wrap along latitude
        max_degrees = 180
        new_coords = np.mod(n + da.coords[dim] + 90, max_degrees) - 90
        da = da.assign_coords({dim: new_coords}).sortby(dim)
        return da.isel(lat=slice(abs(n), -abs(n)))
    else:
        err_msg = (
            'I can only wiggle along {:} or {:} and the respective'
            ' uppercase versions :( Sorry!'
        )
        err_msg = err_msg.format(VALID_LONS_NAMES, VALID_LAT_NAMES)
        raise ValueError(err_msg)


def compute_gradient_along_dim(da, dim, dx):
    '''Compute gradient along a given dimension ``dim`` over a distance ``dx``.

    The distance ``dx`` is the number of steps to be shifted along ``dim``.

    '''
    return wiggle(da, dim, dx // 2) - wiggle(da, dim, -dx // 2)


def compute_amplitude(da1, da2):
    ''' Compute the amplitude of two ``DataArray``

    Instead of using the gradient along a given dimension (lon or lat),
    perhaps the combined amplitude could be more informative (?)
    '''
    return np.sqrt(da1**2 + da2**2)
