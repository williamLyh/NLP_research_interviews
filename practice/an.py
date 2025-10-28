

class CloudFileSystem():
    def __init__(self):
        self.global_storage = {}
        self.user_storage = {'admin': {}}
        self.user_capacity = {}

    def addFile(self, name, size, user='admin'):
        if name in self.global_storage:
            return False
        else:
            self.global_storage[name] = size
            self.user_storage[user][name] = size
            return True
        
    def deleteFile(self, name, user='admin'):
        if name in self.global_storage:
            file_size = self.global_storage.pop(name)
            if user=='admin': self.user_storage['admin'].pop(name)
            # self.user_storage[user].pop(name)
            for user_id, files in self.user_storage.items():
                if name in files:
                    files.pop(name)
                    break
            return file_size
        else:
            return None
        
    def getFileSize(self, name):
        if name in self.global_storage:
            return self.global_storage[name]
        else:
            return None
        
    
    def getNLargePrefixFiles(self, prefix, N):
        matched_files = []
        for name, size in self.global_storage.items():
            if name.startswith(prefix):
                matched_files.append((name, size))
        sorted_files = sorted(matched_files, key=lambda x: (-x[1], x[0]))
        
        output = []
        num_items = min(N, len(sorted_files))
        for i in range(num_items):
            output.append((sorted_files[i][0], sorted_files[i][1]))
        

    def addUser(self, name, capacity):
        if name in self.user_capacity:
            return False
        else:
            self.user_capacity[name] = capacity
            self.user_storage[name] = {}
            return True


    def addFileBy(self, name, size, user_id):
        all_user_fiiles = self.user_storage[user_id]
        user_cur_size = sum(all_user_fiiles.values()) 
    
        if size + user_cur_size > self.user_capacity[user_id]:
            return None
        else:
            self.addFile(name, size, user_id)
            return self.user_capacity[user_id] - (size + user_cur_size)


    def mergeUser(self, user_id_1, user_id_2):
        if (user_id_1 not in self.user_capacity) or \
           (user_id_2 not in self.user_capacity) or \
           (user_id_1 == user_id_2):
            return False
        self.user_capacity[user_id_1] += self.user_capacity.pop(user_id_2)
        self.user_storage[user_id_1].update(self.user_storage[user_id_2])
        self.user_storage.pop(user_id_2)
        return True


    def compressFile(self, user_id, name):
        if name not in self.user_storage[user_id]:
            return None
        file_size = self.deleteFile(name)
        remaining_capacity = self.addFileBy(f'{name}.COMPRESSED', file_size/2, user_id)
        return remaining_capacity
    
    def decompressFile(self, user_id, name):
        suffix = '.COMPRESSED'
        decompressed_name = name[:-len(suffix)]
        file_size = self.getFileSize(name) 
        if decompressed_name in self.global_storage:
            return None
        
        all_user_fiiles = self.user_storage[user_id]
        user_cur_size = sum(all_user_fiiles.values()) 
        if file_size*2 + user_cur_size > self.user_capacity[user_id]:
            return None
        
        self.deleteFile(name)
        remaining_capacity = self.addFileBy(decompressed_name, file_size*2, user_id)
        return remaining_capacity



# COMPRESS_FILE <userId> <name>
# Should compress the file name if it belongs to userId.
# The compressed file should be replaced with a new file named <name>.COMPRESSED.
# The size of the newly created file should be equal to half of the original file. The size of all files is guaranteed to be even, so there should be no fractional calculations.
# It is also guaranteed that name for this operation never points to a compressed file (i.e., it never ends with .COMPRESSED).
# Compressed files should be owned by userId — the owner of the original file.
# Returns a string representing the remaining storage capacity for userId if the file was compressed successfully or an empty string otherwise.


# DECOMPRESS_FILE <userId> <name>
# Should revert the compression of the file name if it belongs to userId.
# It is guaranteed that name for this operation always ends with .COMPRESSED.
# If decompression results in the userId exceeding their storage capacity limit or a decompressed version of the file with the given name already exists, the operation fails.
# Returns a string representing the remaining capacity of userId if the file was decompressed successfully or an empty string otherwise.

# def testcase_1():
#     storage = CloudFileSystem()
#     storage.addFile("file1.txt", 100)
#     storage.addFile("file2.txt", 200)
#     storage.deleteFile("file1.txt")
#     size = storage.getFileSize("file2.txt")
#     print(size)  # Expected output: 200
#     storage.getNLargePrefixFiles("file", 2)  # Expected output: file



# if __name__ == '__main__':
#     testcase_1()


import unittest

# class TestCloudFileSystem(unittest.TestCase):

#     def testcase1(self):
#         fs = CloudFileSystem()
#         self.assertTrue(fs.addFile("file1.txt", 100))


# if __name__ == "__main__":
#     unittest.main()


class TestCloudFileSystem(unittest.TestCase):
    """
    Unit tests for the CloudFileSystem class.
    Each test case creates its own instance of the class.
    """

    def test_add_file_success(self):
        """Test successfully adding a new file."""
        fs = CloudFileSystem()
        self.assertTrue(fs.addFile("file1.txt", 100), "Should add a new file")
        self.assertEqual(fs.getFileSize("file1.txt"), 100, "File size should be correct")
        self.assertIn("file1.txt", fs.user_storage['admin'], "File should be in user's storage")

    def test_add_file_duplicate_name(self):
        """Test that adding a file with an existing name fails."""
        fs = CloudFileSystem()
        fs.addFile("file1.txt", 100)
        self.assertFalse(fs.addFile("file1.txt", 200), "Should not add a duplicate file")
        self.assertEqual(fs.getFileSize("file1.txt"), 100, "Original file size should not change")

    def test_delete_file_success(self):
        """Test successfully deleting an existing file."""
        fs = CloudFileSystem()
        fs.addFileBy("file_to_delete.txt", 50, user_id='testuser')
        deleted_size = fs.deleteFile("file_to_delete.txt")
        self.assertEqual(deleted_size, 50, "Should return the size of the deleted file")
        self.assertIsNone(fs.getFileSize("file_to_delete.txt"), "File should be removed from global storage")
        self.assertNotIn("file_to_delete.txt", fs.user_storage['testuser'], "File should be removed from user's storage")

    def test_delete_file_nonexistent(self):
        """Test that deleting a file that doesn't exist returns None."""
        fs = CloudFileSystem()
        self.assertIsNone(fs.deleteFile("nonexistent.txt"), "Should return None for a nonexistent file")

    def test_add_user(self):
        """Test adding a new user and handling duplicate user additions."""
        fs = CloudFileSystem()
        self.assertTrue(fs.addUser("bob", 1000), "Should successfully add a new user")
        self.assertIn("bob", fs.user_capacity, "User should have a capacity entry")
        self.assertIn("bob", fs.user_storage, "User should have a storage entry")
        self.assertFalse(fs.addUser("bob", 500), "Should not add an existing user")
        self.assertEqual(fs.user_capacity["bob"], 1000, "User capacity should not be overwritten")

    def test_add_file_by_user_within_capacity(self):
        """Test adding a file that is within the user's storage capacity."""
        fs = CloudFileSystem()
        fs.addUser("alice", 500)
        remaining = fs.addFileBy("data.csv", 200, "alice")
        self.assertEqual(remaining, 300, "Should return correct remaining capacity")
        self.assertEqual(fs.getFileSize("data.csv"), 200, "File should be created")

    def test_add_file_by_user_exceeds_capacity(self):
        """Test that adding a file fails if it exceeds the user's capacity."""
        fs = CloudFileSystem()
        fs.addUser("alice", 500)
        fs.addFileBy("data1.csv", 400, "alice")
        result = fs.addFileBy("data2.csv", 150, "alice")
        self.assertIsNone(result, "Should return None when capacity is exceeded")
        self.assertIsNone(fs.getFileSize("data2.csv"), "File exceeding capacity should not be created")

    def test_get_n_large_prefix_files(self):
        """Test retrieving the largest files matching a prefix."""
        fs = CloudFileSystem()
        fs.addFile("img_01.jpg", 100)
        fs.addFile("img_03.jpg", 300)
        fs.addFile("doc_a.txt", 50)
        fs.addFile("img_02_large.jpg", 500)
        fs.addFile("img_04_small.jpg", 20) # Same prefix, smaller size
        fs.addFile("img_00_also_large.jpg", 500) # Same size, different name

        # Expected sort order:
        # 1. img_00_also_large.jpg (500) - name comes first alphabetically
        # 2. img_02_large.jpg (500)
        # 3. img_03.jpg (300)
        # 4. img_01.jpg (100)
        # 5. img_04_small.jpg (20)
        expected_top_3 = [("img_00_also_large.jpg", 500), ("img_02_large.jpg", 500), ("img_03.jpg", 300)]
        result = fs.getNLargePrefixFiles("img_", 3)
        self.assertEqual(result, expected_top_3, "Should return the top 3 largest files, sorted correctly")

        # Test with N larger than matches
        self.assertEqual(len(fs.getNLargePrefixFiles("img_", 10)), 5, "Should return all matched files if N is large")

        # Test with no matches
        self.assertEqual(len(fs.getNLargePrefixFiles("vid_", 5)), 0, "Should return empty list for no matches")

    def test_merge_user_success(self):
        """Test successfully merging one user into another."""
        fs = CloudFileSystem()
        fs.addUser("user1", 1000)
        fs.addFileBy("file_a.txt", 100, "user1")

        fs.addUser("user2", 2000)
        fs.addFileBy("file_b.txt", 200, "user2")

        self.assertTrue(fs.mergeUser("user1", "user2"), "Merge should be successful")
        self.assertEqual(fs.user_capacity["user1"], 3000, "Capacities should be merged")
        self.assertNotIn("user2", fs.user_capacity, "Merged user's capacity should be removed")
        self.assertNotIn("user2", fs.user_storage, "Merged user's storage should be removed")
        self.assertIn("file_a.txt", fs.user_storage["user1"], "Original files should remain")
        self.assertIn("file_b.txt", fs.user_storage["user1"], "New files should be added")

    def test_merge_user_invalid(self):
        """Test merge failures for non-existent or identical users."""
        fs = CloudFileSystem()
        fs.addUser("user1", 1000)
        self.assertFalse(fs.mergeUser("user1", "nonexistent_user"), "Should fail if user2 doesn't exist")
        self.assertFalse(fs.mergeUser("nonexistent_user", "user1"), "Should fail if user1 doesn't exist")
        self.assertFalse(fs.mergeUser("user1", "user1"), "Should fail if users are the same")



# queries = [
#     ["ADD_USER", "user1", "1000"],
#     ["ADD_USER", "user2", "5000"], 
#     ["ADD_FILE_BY", "user1", "/dir/file.mp4", "500"],
#     ["ADD_FILE_BY", "user2", "/dir/file.mp4", "1"],
#     ["COMPRESS_FILE", "user3", "/dir/file.mp4"],
#     ["COMPRESS_FILE", "user1", "/folder/non_existing_file"],
#     ["COMPRESS_FILE", "user1", "/dir/file.mp4"],
#     ["COMPRESS_FILE", "user1", "/dir/file.mp4"],
#     ["GET_FILE_SIZE", "/dir/file.mp4.COMPRESSED"],
#     ["GET_FILE_SIZE", "/dir/file.mp4"],
#     ["COPY_FILE", "/dir/file.mp4.COMPRESSED", "/file.mp4.COMPRESSED"],
#     ["ADD_FILE_BY", "user1", "/dir/file.mp4", "500"],
#     ["DECOMPRESS_FILE", "user1", "/dir/file.mp4.COMPRESSED"],
#     ["UPDATE_CAPACITY", "user1", "2000"],
#     ["DECOMPRESS_FILE", "user2", "/dir/file.mp4.COMPRESSED"],
#     ["DECOMPRESS_FILE", "user3", "/dir/file.mp4.COMPRESSED"],
#     ["DECOMPRESS_FILE", "user1", "/dir/file.mp4.COMPRESSED"]
# ]
# The output should be:
# ["true", "true", "500", "", "", "", "750", "", "250", "", "true", "0", "0", "true", "", "", "750"]

    def test_compress_file_success(self):
        fs = CloudFileSystem()
        self.assertTrue(fs.addUser("user1", 1000))
        self.assertTrue(fs.addUser("user2", 5000))
        self.assertEqual(fs.addFileBy("/dir/file.mp4", 500, "user1"), 500)
        # self.assertEqual(fs.addFileBy("/dir/file.mp4", 1, "user

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
