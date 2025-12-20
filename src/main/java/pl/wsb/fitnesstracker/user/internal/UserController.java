package pl.wsb.fitnesstracker.user.internal;

import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;
import pl.wsb.fitnesstracker.user.api.User;
import pl.wsb.fitnesstracker.user.api.UserDto;
import pl.wsb.fitnesstracker.user.api.UserEmailDto;

import java.time.LocalDate;
import java.util.List;

/**
 * UserController is responsible for handling HTTP requests related to user operations.
 * It provides endpoints for retrieving, creating, updating and deleting users.
 * 
 * This is a REST API controller that supports the following operations:
 * - List all users or get users with specific criteria
 * - Get user details by ID or email
 * - Create new users
 * - Update existing users
 * - Delete users
 * 
 * All endpoints return data in JSON format and use appropriate HTTP status codes.
 */
@RestController   //// 8. Automatic to JSON
@RequestMapping("/v1/users")
@RequiredArgsConstructor
class UserController {

    private final UserServiceImpl userService;

    private final UserMapper userMapper;

    ///// TEST
    /**
     * Retrieves all users with full details (ID, firstName, lastName, birthdate, email).
     * 
     * @return list of all users in the system with complete information
     * @see UserDto
     */
    @GetMapping
    public List<UserDto> getAllUsers() {
        return userService.findAllUsers()
                .stream()
                .map(userMapper::toDto)
                .toList();
    }


    /// 1. 
    /**
     * Retrieves all users with simple details (firstName and lastName only).
     * Used for listing users without sensitive information.
     * 
     * @return list of all users with basic information
     * @see UserDto
     */
    @GetMapping("/simple")
    public List<UserDto> getSimpleUsers() {
        return userService.findAllUsers()
                .stream()
                .map(userMapper::toDto)
                .toList();
    }

    /// 2.
    /**
     * Retrieves a specific user by their ID.
     * 
     * @param id the unique identifier of the user to retrieve
     * @return the user details if found
     * @throws RuntimeException if user with given ID is not found (404)
     * @see UserDto
     */
    @GetMapping("/{id}")
    public UserDto getUserById(@PathVariable Long id) {
        return userService.getUser(id)
                .map(userMapper::toDto)
                .orElseThrow(() -> new RuntimeException("User not found"));
    }

    /// 3.
    /**
     * Creates a new user in the system.
     * 
     * @param userDto the user data transfer object containing user details
     * @return the created user with assigned ID
     * @throws IllegalArgumentException if user data is invalid
     * @see UserDto
     */
    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public UserDto createUser(@RequestBody UserDto userDto) {
        User user = new User(userDto.firstName(), userDto.lastName(), userDto.birthdate(), userDto.email());
        User createdUser = userService.createUser(user);
        return userMapper.toDto(createdUser);
    }

    /// 4.
    /**
     * Deletes a user from the system.
     * 
     * @param userId the unique identifier of the user to delete
     * @return no content (204 No Content)
     */
    @DeleteMapping("/{userId}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteUser(@PathVariable Long userId) {
        userService.deleteUser(userId);
    }

    /// 5.
    /**
     * Searches for users by email address.
     * Returns only user ID and email address.
     * 
     * @param email the email address to search for
     * @return list containing the user's ID and email if found, empty list otherwise
     * @see UserEmailDto
     */
    @GetMapping("/email")
    public List<UserEmailDto> getUserByEmail(@RequestParam String email) {
        return userService.getUserByEmail(email)
                .map(user -> List.of(userMapper.toEmailDto(user)))
                .orElse(List.of());
    }

    /// 6.
    /**
     * Retrieves all users who are older than the specified date.
     * Compares birth dates to determine age.
     * 
     * @param time the threshold date - returns users born before this date
     * @return list of users older than the specified date
     * @see UserDto
     */
    @GetMapping("/older/{time}")
    public List<UserDto> getUsersOlderThan(@PathVariable LocalDate time) {
        return userService.findAllUsers()
                .stream()
                .filter(user -> user.getBirthdate().isBefore(time))
                .map(userMapper::toDto)
                .toList();
    }

    /// 7.
    /**
     * Updates an existing user with new information.
     * 
     * @param userId the unique identifier of the user to update
     * @param userDto the new user data
     * @return the updated user information
     * @throws RuntimeException if user with given ID is not found (404)
     * @see UserDto
     */
    @PutMapping("/{userId}")
    public UserDto updateUser(@PathVariable Long userId, @RequestBody UserDto userDto) {
        userService.getUser(userId)
                .orElseThrow(() -> new RuntimeException("User not found"));
        User updatedUser = new User(userDto.firstName(), userDto.lastName(), userDto.birthdate(), userDto.email());
        updatedUser.setId(userId);
        userService.updateUser(updatedUser);
        return userMapper.toDto(updatedUser);
    }
}

