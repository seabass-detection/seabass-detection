# pylint: disable=C0302, line-too-long, too-few-public-methods,
# too-many-branches, too-many-statements, no-member, ungrouped-imports,
# too-many-arguments, wrong-import-order, relative-import,
# too-many-instance-attributes, too-many-locals, unused-variable,
# not-context-manager
'''deals with database operations related to lenscorrecton.py'''

# region imports
import sqlite3 as _sqlite3
import fuckit as _fuckit
import pickle as _pickle

import funclib.baselib as _baselib
# endregion


#Coped from dblib.sqlitelib
class Conn(object):
    '''connection to database'''

    def __init__(self, cnstr=':memory:'):
        self.cnstr = cnstr
        self.conn = None

    def __enter__(self):
        self.conn = _sqlite3.connect(self.cnstr)
        self.conn.row_factory = _sqlite3.Row
        return self.conn

    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

    def open(self, cnstr):
        '''str->void
        file path location of the db
        open a connection, closing existing one
        '''
        self.close()
        self.conn = _sqlite3.connect(cnstr)
        self._conn.row_factory = _sqlite3.Row

    def close(self, commit=False):
        '''close the db'''
        with _fuckit:
            if commit:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

    def commit(self):
        '''commit'''
        self._conn.commit()


class CalibrationCRUD(object):
    '''everything to do with the db
    '''

    def __init__(self, dbconn):
        '''(cnstr)
        pass in an open dbconn
        '''
        self.conn = dbconn
        assert isinstance(self.conn, _sqlite3.Connection)

    # region exists stuff
    def exists_by_primarykey(self, table, keyid):
        '''(str,id)->bool
        return bool indicating if  id exists in table
        '''
        cur = self.conn.cursor()
        sql = 'SELECT EXISTS(SELECT 1 as one FROM ' + table + ' WHERE ' + \
            table + 'id="' + str(keyid) + '" LIMIT 1) as res;'
        cur.execute(sql)
        row = cur.fetchall()
        for res in row:
            return bool(res)

    def exists_by_compositekey(self, table, dic):
        '''(str, dic)->bool
        Return true or false if a record exists in table based on multiple values
        so dic would be for e.g.
        foreignkey1.id=1, foreignkey2.id=3, id=5
        '''
        sql = []
        where = ["%s='%s' AND " % (j, k) for j, k in dic.items()]
        where[-1] = where[-1].replace('AND', '')
        sql.append('select  exists(select 1 from %s where ' % (table))
        sql.append("".join(where))
        sql.append('LIMIT 1);')
        query = "".join(sql)
        cur = self.conn.cursor()
        cur.execute(query)
        row = cur.fetchall()
        return bool(row[0][0])
    # endregion


    def get_value(self, table, col_to_read, key_cols):
        '''(str, str, dic:str) -> str|None
        Read the value in col_to_read which matches
        the values assigned to the col-value pairs in key_cols.
        Returns None if no match

        table:
            the name of the table
        col_to_read:
            the name of the column which contains the value to return
        key_cols:
            a dictionary of key value pairs, e.g. {'id':1, 'country':'UK'}

        returns:
            The first matched value, or None

        Exmaple:
            tot = get_value('orders', 'total', {'company':'Amazon', 'region':'UK'})
        '''
        sql = []
        where = ["%s='%s' AND " % (j, k) for j, k in key_cols.items()]
        where[-1] = where[-1].replace('AND', '')
        sql.append('select %s from %s where ' % (col_to_read, table))
        sql.append("".join(where))
        sql.append('LIMIT 1;')
        query = "".join(sql)
        cur = self.conn.cursor()
        cur.execute(query)
        row = cur.fetchall()

        try:
            s = str(row[0][0])
        except Exception:
            s = None

        return s

    # region correction.db specific
    def crud_camera_upsert(self, camera_model):
        '''(str)-> int
        Add a camera

        camera_model:
            Camera model name (e.g. 'GoPro5')

        Returns:
            camera_modelid (int) of added record
        '''
        keys = {'camera_model': camera_model}
        sql = self._sql_upsert('camera_model', keys)
        self.executeSQL(sql)

        i = self.get_value('camera_model', 'camera_modelid', {'camera_model':camera_model})

        return i

    def crud_calibration_upsert(
            self,
            camera_modelid,
            width,
            height,
            camera_matrix,
            dcoef,
            rms,
            rvect,
            tvect,
            K,
            D):
        '''update/insert calibration record
        
        Returns:
            calibrationid of updated/inserted row
        '''
        keys = {
            'camera_modelid': camera_modelid,
            'width': width,
            'height': height}
        sql = self._sql_upsert(
            'calibration',
            keys,
            width=width,
            height=height,
            rms=rms,
            timestamp='CURRENT_TIMESTAMP'
            )
        self.executeSQL(sql)

        calibrationid = self.get_value('calibration', 'calibrationid', keys)
        self._blobs(calibrationid, camera_matrix, dcoef, rvect, tvect, K, D)
        return calibrationid

    def crud_calibration_delete_by_composite(self, id_or_name, height, width):
        '''(int|str, int, int)
        Delete calibration by unique key
        '''
        assert isinstance(id_or_name, (int, str)), 'id_or_name should be a string (the name) or the camera id from the db'
        
        if isinstance(id_or_name, str):
            i = self.get_value('camera_model', 'camera_modelid', {'camera_model':id_or_name})
            if i is None:
                return
        else:
            i = int(id_or_name)

        sql = CalibrationCRUD._sql_delete(
            'calibration',
            camera_modelid=i,
            height=height,
            width=width)

        self.executeSQL(sql)


    def crud_read_calibration_blobs(self, camera_model, height, width):
        '''(str, int, int)-> dic
        1) Using the key for table calibrtion
        2) Returns the calibration blobs from the db as a dictionary, K and D are for fisheye calibrations
        3) {'cmat':cmat, 'dcoef':dcoef, 'rvect':rvect, 'tvect':tvect, 'K':K, 'D':D}
        4) Returns None if no match.
        '''
        cameramodelid = self._lookup(
            'camera_model',
            'camera_model',
            'camera_modelid',
            camera_model)
        if cameramodelid is None:
            raise ValueError(
                'cameramodelid could not be read for %s' %
                camera_model)

        sql = CalibrationCRUD._sql_read(
            'calibration',
            camera_modelid=cameramodelid,
            height=height,
            width=width)
        cur = self.conn.cursor()
        res = cur.execute(sql)
        assert isinstance(res, _sqlite3.Cursor)
        for row in res:
            if not row:
                return None
            iif = lambda x: None if x is None else _pickle.loads(x) 
            cmat = iif(row['camera_matrix'])
            dcoef = iif(row['distortion_coefficients'])
            rvect = iif(row['rotational_vectors'])
            tvect = iif(row['translational_vectors'])
            K = iif(row['K'])
            D = iif(row['D'])
            return {
                'cmat': cmat,
                'dcoef': dcoef,
                'rvect': rvect,
                'tvect': tvect,
                'K': K,
                'D': D
                }

    def executeSQL(self, sql):
        '''execute sql against the db'''
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def _blobs(self, calibrationid, camera_matrix, dcoef, rvect, tvect, K, D):
        '''add the blobs seperately, easier because we let upsert generator deal with the insert/update
        but then just pass the composite key to edit the record
        '''
        cur = self.conn.cursor()
        cm_b = _sqlite3.Binary(camera_matrix)
        dcoef_b = _sqlite3.Binary(dcoef)
        rvect_b = _sqlite3.Binary(rvect)
        tvect_b = _sqlite3.Binary(tvect)

        #K_b is null if we havent asked for a fisheye lens correction
        if K is None or D is None:
            sql = 'UPDATE calibration SET camera_matrix=?, distortion_coefficients=?, rotational_vectors=?, translational_vectors=?' \
                ' WHERE calibrationid=?'
            cur.execute(sql, (cm_b, dcoef_b, rvect_b, tvect_b, calibrationid))
        else:
            K_b = _sqlite3.Binary(K)
            D_b = _sqlite3.Binary(D)
            sql = 'UPDATE calibration SET camera_matrix=?, distortion_coefficients=?, rotational_vectors=?, translational_vectors=?, K=?, D=?' \
                ' WHERE calibrationid=?'
            cur.execute(sql, (cm_b, dcoef_b, rvect_b, tvect_b, K_b, D_b, calibrationid))


    def list_existing(self):
        '''void->list
        Lists all available profiles
        '''
        sql = 'select' \
            ' camera_model || ": " || cast(width as text) || "x" || cast(height as text) ||' \
            ' case' \
            '    when K is null and camera_matrix is null then " No Standard, No Fisheye  " || cast(timestamp as text)' \
            '    when K is null and camera_matrix is not null then " Standard, No Fisheye  " || cast(timestamp as text)' \
            '    when K is not null and camera_matrix is null then " No Standard, Fisheye  " || cast(timestamp as text)' \
            '    else " Standard, Fisheye  " || cast(timestamp as text)' \
            ' end as res' \
            ' from' \
            ' camera_model inner join calibration on camera_model.camera_modelid=calibration.camera_modelid' \
            ' order by' \
            ' camera_model,' \
            ' cast(width as text) || "x" || cast(height as text)'
        
        cur = self.conn.cursor()
        res = cur.execute(sql)
        assert isinstance(res, _sqlite3.Cursor)
        ret = []
        for row in res:
            if not row:
                return None
            ret.append(row['res'])
        return ret


    def list_param(self, camera, x, y, param):
        '''str, int, int, str->list
        Lists all available profiles
        '''
        s = []
        sql = ''
        s.append('select %s as res' % param)
        s.append(' from camera_model inner join calibration on camera_model.camera_modelid=calibration.camera_modelid')
        s.append(' where camera_model=? and width=? and height=?') 
        sql = ''.join(s)

        cur = self.conn.cursor()
        res = cur.execute(sql, [camera, x, y])
        assert isinstance(res, _sqlite3.Cursor)
        for row in res:
            if not row:
                return None
            ret = _pickle.loads(row['res'])
            break
        return ret


    def blobs_get_nearest_aspect_match(self, camera_model, height, width):
        '''(str, int, int)->dict
        Returns the calibration matrices for the nearest matching
         aspect for which we have correction matrices
        {'cmat':cmat, 'dcoef':dcoef, 'rvect':rvect, 'tvect':tvect, 'K':K, 'D':D, 'matched_resolution_w_by_h':(w,h), 'new_aspect':w/h)}
        '''

        aspect = width / float(height)
        sql = 'SELECT calibration.height, calibration.width, calibration.camera_matrix, calibration.distortion_coefficients,' \
            ' calibration.rotational_vectors, calibration.translational_vectors, calibration.K, calibration.D,' \
            ' width/cast(height as float) as aspect' \
            ' FROM calibration' \
            ' INNER JOIN camera_model ON calibration.camera_modelid=camera_model.camera_modelid' \
            ' WHERE camera_model.camera_model=?' \
            ' ORDER BY ABS(? - (width/cast(height as float))) LIMIT 1'
        cur = self.conn.cursor()
        res = cur.execute(sql, [camera_model, aspect])
        assert isinstance(res, _sqlite3.Cursor)
        for row in res:
            if not row:
                return None

            cmat = _pickle.loads(row['camera_matrix'])
            dcoef = _pickle.loads(row['distortion_coefficients'])
            rvect = _pickle.loads(row['rotational_vectors'])
            tvect = _pickle.loads(row['translational_vectors'])
            K = _pickle.loads(row['K'])
            D = _pickle.loads(row['D'])
            w = row['width']
            h = row['height']
            aspect = w / float(h)
            return {
                'cmat': cmat,
                'dcoef': dcoef,
                'rvect': rvect,
                'tvect': tvect,
                'K': K,
                'D': D,
                'matched_resolution_w_by_h': (
                    w,
                    h),
                'new_aspect': aspect}
    # endregion

    # helpers
    def _get_last_id(self, table_name):
        '''str->int
        1) Returns last added id in table table_name
        2) Returns None if no id
        '''
        sql = 'select seq from sqlite_sequence where name="%s"' % table_name
        cur = self.conn.cursor()
        cur.execute(sql)
        row = cur.fetchall()
        return CalibrationCRUD._read_col(row, 'seq')

    @staticmethod
    def _read_col(cur, colname):
        '''cursor, str->basetype
        reads a cursor row column value
        returns None if there is no row
        First row only
        '''
        if not cur:
            return None
        else:
            for results in cur:
                return results[colname]

    def _lookup(
            self,
            table_name,
            col_to_search,
            col_with_value_we_want,
            value):
        '''(str, str, str, basetype)->basetype
        1) Returns lookup value, typically based on a primary key value
        2) Returns None if no matches found
        '''
        sql = 'SELECT %s FROM %s WHERE %s="%s" LIMIT 1;' % \
            (col_with_value_we_want, table_name, col_to_search, value)
        cur = self.conn.cursor()
        cur.execute(sql)
        row = cur.fetchall()
        return self._read_col(row, col_with_value_we_want)

    @staticmethod
    def _sql_read(table, **kwargs):
        ''' Generates SQL for a SELECT statement matching the kwargs passed. '''
        sql = list()
        sql.append("SELECT * FROM %s " % table)
        if kwargs:
            sql.append("WHERE " + " AND ".join("%s = '%s'" % (k, v)
                                               for k, v in kwargs.items()))
        sql.append(";")
        return "".join(sql)

    def _sql_upsert(self, table, keylist, **kwargs):
        '''(str, dict, **kwargs)->str
        Generate an SQL statement to either insert or update an sql table.
        If the record exists, as determined by the keylist then an update is
        generated, otherwise an insert
        
        Keylist is dictionary of key fields and their values used to build the where.
        
        Field values are passed as kwargs.

        The upsert will considers a value string of 'CURRENT_TIMESTAMP'
        a special case, and will strip the quotes so the corresponding
        field gets set to CURRENT_TIMESTAMP. e.g. timestamp='CURRENT_TIMESTAMP'
        will be timestamp=CURRENT_TIMESTAMP in the final sql.

        table:
            Database table name
        keylist:
            Builds the where e.g. {'orderid':1, 'supplier':'Widget Company'}
        kwargs:
            Fields to insert/update

        returns:
            The insert or update SQL as a string
        '''
        allargs = _baselib.dic_merge_two(keylist, kwargs)
        sql_insert = []
        sql_update = []
        if self.exists_by_compositekey(table, keylist):
            where = [" %s='%s' " % (j, k) for j, k in keylist.items()]

            update = ["%s='%s'" % (j, k) for j, k in allargs.items()]

            sql_update.append("UPDATE %s SET " % (table))
            sql_update.append(", ".join(update))
            sql_update.append(" WHERE %s" % (" AND ".join(where)))

            ret = "".join(sql_update)
            ret = ret.replace("'CURRENT_TIMESTAMP'", "CURRENT_TIMESTAMP")
            return ret


        keys = ["%s" % k for k in allargs]
        values = ["'%s'" % v for v in allargs.values()]
        sql_insert = list()
        sql_insert.append("INSERT INTO %s (" % table)
        sql_insert.append(", ".join(keys))
        sql_insert.append(") VALUES (")
        sql_insert.append(", ".join(values))
        sql_insert.append(");")
        ret = "".join(sql_insert)
        ret = ret.replace("'CURRENT_TIMESTAMP'", "CURRENT_TIMESTAMP")
        return ret

    @staticmethod
    def _sql_delete(table, **kwargs):
        '''(str, dict) -> str
        Generates a delete sql from keyword/value pairs
        where keyword is the column name and value is the value to match.

        table:
            table name
        kwargs:
            Key/value pairs, e.g. {'camera':'GoPro', 'x':1024, 'y':768}

        returns:
            The delete SQL
        '''
        sql = list()
        sql.append("DELETE FROM %s " % table)
        sql.append("WHERE " + " AND ".join("%s = '%s'" % (k, v)
                                           for k, v in kwargs.items()))
        sql.append(";")
        return "".join(sql)


def main():
    '''run when executed directly'''

    pass


# This only executes if this script was the entry point
if __name__ == '__main__':
    main()
    # execute my code
