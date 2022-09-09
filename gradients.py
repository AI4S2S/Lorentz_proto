import numpy as np


def _wrapAngle360(lon):
    '''wrap angle to `[0, 360[`.'''
    lon = np.array(lon)
    return np.mod(lon, 360)


def _wrapAngle180(lon):
    '''wrap angle to `[-180, 180[`.'''
    lon = np.array(lon)
    sel = (lon < -180) | (180 <= lon)
    lon[sel] = _wrapAngle360(lon[sel] + 180) - 180
    return lon


def _is_180(lon_min, lon_max, msg_add=''):

    lon_min = np.round(lon_min, 6)
    lon_max = np.round(lon_max, 6)

    if (lon_min < 0) and (lon_max > 180):
        msg = 'lon has both data that is larger than 180 and smaller than 0. ' + msg_add
        raise ValueError(msg)

    return lon_max <= 180


def _wrapAngle(lon, wrap_lon=True, is_unstructured=False):
    '''wrap the angle to the other base
    If lon is from -180 to 180 wraps them to 0..360
    If lon is from 0 to 360 wraps them to -180..180
    '''

    if np.isscalar(lon):
        lon = [lon]

    lon = np.array(lon)
    new_lon = lon

    if wrap_lon is True:
        _is_180(lon.min(), lon.max(), msg_add='Cannot infer the transformation.')

    wl = int(wrap_lon)

    if wl == 180 or (wl != 360 and lon.max() > 180):
        new_lon = _wrapAngle180(lon.copy())

    if wl == 360 or (wl != 180 and lon.min() < 0):
        new_lon = _wrapAngle360(lon.copy())

    # check if they are still unique
    if new_lon.ndim == 1 and not is_unstructured:
        if new_lon.shape != np.unique(new_lon).shape:
            raise ValueError('There are equal longitude coordinates (when wrapped)!')

    return new_lon


def _wiggle(da, dim, n):
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
        lons = da.coords[dim]
        # If lons are from -180..180 wrap them to 0..360
        if _is_180(lons.min(), lons.max()):
            new_lons = _wrapAngle(lons, wrap_lon=True)
            da = da.assign_coords({dim: new_lons})
        max_degrees = 360
        new_coords = np.mod(n + da.coords[dim], max_degrees)
        da = da.assign_coords({dim: new_coords}).sortby(dim)

        return da
    elif dim.lower() in VALID_LAT_NAMES:
        # Wrap along latitude
        max_degrees = 180
        new_coords = np.mod(n + da.coords[dim] + 90, max_degrees) - 90
        da = da.assign_coords({dim: new_coords}).sortby(dim)
        return da.isel({dim: slice(abs(n), -abs(n))})
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
    return _wiggle(da, dim, dx // 2) - _wiggle(da, dim, -dx // 2)


def compute_amplitude(da1, da2):
    ''' Compute the amplitude of two ``DataArray``

    Instead of using the gradient along a given dimension (lon or lat),
    perhaps the combined amplitude could be more informative (?)
    '''
    return np.sqrt(da1**2 + da2**2)
